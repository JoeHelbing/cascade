"""
Resistance Cascade ABM - Block-per-Simulation GPU Kernel.

Architecture: one thread block per simulation, multiple threads per block.
Each thread handles ceil(n_agents/BLOCK_SIZE) agents in parallel.
barrier() synchronizes between simulation phases.

Optimizations over one-thread-per-sim:
- Parallel agent processing within each step (256-way parallelism)
- O(nearby) spatial grid neighbor lookup (replaces O(n_agents) full scan)
- L1 cache benefits from all threads reading same agent data
- Variable vision parameter (runtime, not comptime)
"""

from std.sys import has_accelerator, argv
from std.math import exp, log, sqrt
from std.collections import List
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx, block_dim, global_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.os.atomic import Atomic


# ============================================================
# Constants
# ============================================================
comptime SUPPORT: Int32 = 0
comptime ACTIVE: Int32 = 1
comptime OPPOSE: Int32 = 2
comptime JAILED: Int32 = 3
comptime SECURITY_COND: Int32 = 4

# Int8 versions for shared memory comparisons
comptime SUPPORT_I8: Int8 = 0
comptime ACTIVE_I8: Int8 = 1
comptime OPPOSE_I8: Int8 = 2
comptime JAILED_I8: Int8 = 3

comptime GRID_W: Int = 33
comptime GRID_H: Int = 33
comptime GRID_SIZE: Int = GRID_W * GRID_H  # 1089 cells

comptime MAX_AGENTS: Int = 1024  # Fits in shared memory (1024 * ~56B = 56KB < 100KB)
comptime BLOCK_SIZE: Int = 256   # Threads per block (each handles ~4 agents)
comptime MAX_PER_CELL: Int = 8   # Max agents per grid cell
comptime MAX_STEPS: Int = 500    # Max steps per simulation (for step_metrics buffer sizing)
comptime MAX_TRACE_STEPS: Int = MAX_STEPS + 1  # Step 0 initial state + each executed step
comptime N_STEP_FIELDS: Int = 5  # Per-step model metrics: active, support, oppose, jail, revolution


# ============================================================
# LCG random number generator (deterministic, GPU-safe)
# ============================================================

@always_inline
fn lcg_next(state: UInt64) -> UInt64:
    return state * 6364136223846793005 + 1442695040888963407


@always_inline
fn lcg_float(state: UInt64) -> Float32:
    return Float32(Int64((state >> 33) & 0x7FFFFFFF)) / Float32(2147483648.0)


@always_inline
fn lcg_int(state: UInt64, max_val: Int) -> Int:
    return Int((state >> 33) % UInt64(max_val))


fn lcg_gauss_val(state: UInt64, mean: Float32, std: Float32) -> Tuple[Float32, UInt64]:
    """Marsaglia polar method: generates one Gaussian sample.
    Returns (value, new_state). No trig needed -- GPU compatible."""
    var s = state
    var v1: Float32 = 0
    var rsq: Float32 = 0
    # Rejection loop: ~78.5% acceptance rate, typically 1-2 iterations
    while True:
        s = lcg_next(s)
        v1 = lcg_float(s) * 2.0 - 1.0
        s = lcg_next(s)
        var v2 = lcg_float(s) * 2.0 - 1.0
        rsq = v1 * v1 + v2 * v2
        if rsq < 1.0 and rsq > 0.0:
            break
    var fac = sqrt(Float32(-2.0) * log(rsq) / rsq)
    return (mean + std * v1 * fac, s)


@always_inline
fn sigmoid_f32(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


def cond_name(c: Int32) -> String:
    if c == ACTIVE:
        return String("Active")
    elif c == OPPOSE:
        return String("Oppose")
    elif c == JAILED:
        return String("Jailed")
    elif c == SECURITY_COND:
        return String("Security")
    return String("Support")


# ============================================================
# GPU Kernel: One block per simulation
# ============================================================

def block_sim_kernel(
    # Per-sim params: [sim_id * 9 + param_idx]
    # 0=seed, 1=cit_dens, 2=sec_dens, 3=pp_mean, 4=eps, 5=threshold,
    # 6=max_jail, 7=n_steps, 8=vision
    params: UnsafePointer[Float32, MutAnyOrigin],
    # Per-agent arrays: [sim_id * MAX_AGENTS + agent_id]
    cond: UnsafePointer[Int32, MutAnyOrigin],
    next_cond: UnsafePointer[Int32, MutAnyOrigin],
    pos_x: UnsafePointer[Int32, MutAnyOrigin],
    pos_y: UnsafePointer[Int32, MutAnyOrigin],
    is_citizen: UnsafePointer[Int32, MutAnyOrigin],
    private_pref: UnsafePointer[Float32, MutAnyOrigin],
    eps_arr: UnsafePointer[Float32, MutAnyOrigin],
    eps_prob_arr: UnsafePointer[Float32, MutAnyOrigin],
    oppose_th: UnsafePointer[Float32, MutAnyOrigin],
    active_th: UnsafePointer[Float32, MutAnyOrigin],
    jail_sent: UnsafePointer[Int32, MutAnyOrigin],
    activation_val: UnsafePointer[Float32, MutAnyOrigin],
    rng_arr: UnsafePointer[UInt64, MutAnyOrigin],
    # Per-sim counts
    num_citizens_arr: UnsafePointer[Int32, MutAnyOrigin],
    num_agents_arr: UnsafePointer[Int32, MutAnyOrigin],
    # Spatial grid: [sim_id * GRID_SIZE + cell] for counts
    # [sim_id * GRID_SIZE * MAX_PER_CELL + cell * MAX_PER_CELL + slot] for cells
    grid_counts: UnsafePointer[Int32, MutAnyOrigin],
    grid_cells: UnsafePointer[Int32, MutAnyOrigin],
    # Output metrics: [sim_id * 6 + metric_idx]
    metrics: UnsafePointer[Int32, MutAnyOrigin],
    # Per-step model metrics: [sim_id * MAX_STEPS * N_STEP_FIELDS + step * N_STEP_FIELDS + field]
    # Fields: 0=active, 1=support, 2=oppose, 3=jail, 4=revolution
    # Pass null pointer to skip per-step collection
    step_metrics: UnsafePointer[Int32, MutAnyOrigin],
    # Per-agent state trace: [sim_id * MAX_TRACE_STEPS * MAX_AGENTS + step * MAX_AGENTS + agent_id]
    trace_cond: UnsafePointer[Int32, MutAnyOrigin],
    trace_pos_x: UnsafePointer[Int32, MutAnyOrigin],
    trace_pos_y: UnsafePointer[Int32, MutAnyOrigin],
    num_sims: Int,
):
    var sid = Int(block_idx.x)
    if sid >= num_sims:
        return
    var tid = Int(thread_idx.x)

    var off = sid * MAX_AGENTS
    var poff = sid * 9

    var n_citizens = Int(num_citizens_arr[sid])
    var n_agents = Int(num_agents_arr[sid])
    var vision = Int(params[poff + 8])
    var threshold_val = params[poff + 5]
    var threshold_sig = sigmoid_f32(threshold_val)
    var max_jail = Int(params[poff + 6])
    var n_steps = Int(params[poff + 7])

    # Spatial grid offsets for this simulation
    var gc_off = sid * GRID_SIZE  # grid_counts offset
    var gcl_off = sid * GRID_SIZE * MAX_PER_CELL  # grid_cells offset

    # ---- Shared memory for hot-path reads (Phase 1 neighbor scan) ----
    # s_cond: random-access per neighbor (~100 reads/agent)
    # s_grid_counts: sequential-access per cell (225 reads/agent at vision=7)
    # Note: is_citizen check replaced by j < n_citizens (static layout)
    # Condition values 0-4 fit in int8 (saves 3KB: 4KB -> 1KB)
    var s_cond = stack_allocation[
        MAX_AGENTS, Scalar[DType.int8], address_space=AddressSpace.SHARED,
    ]()
    var s_grid_counts = stack_allocation[
        GRID_SIZE, Scalar[DType.int16], address_space=AddressSpace.SHARED,
    ]()
    # Per-cell security count: sum over vision to get total security in neighborhood
    var s_sec_counts = stack_allocation[
        GRID_SIZE, Scalar[DType.int16], address_space=AddressSpace.SHARED,
    ]()
    # grid_cells in shared memory: 1089 * 8 * 2 = 17KB (int16 sufficient for 0..761)
    var s_grid_cells = stack_allocation[
        GRID_SIZE * MAX_PER_CELL, Scalar[DType.int16], address_space=AddressSpace.SHARED,
    ]()

    # Use metrics[moff + 4] as revolution flag (global memory, visible to all threads)
    var moff = sid * 6
    if tid == 0:
        metrics[moff + 4] = 0  # revolution flag = false

    barrier()

    if trace_cond:
        var tr_i0 = tid
        while tr_i0 < n_agents:
            var tr_idx0 = sid * MAX_TRACE_STEPS * MAX_AGENTS + tr_i0
            trace_cond[tr_idx0] = cond[off + tr_i0]
            trace_pos_x[tr_idx0] = pos_x[off + tr_i0]
            trace_pos_y[tr_idx0] = pos_y[off + tr_i0]
            tr_i0 += BLOCK_SIZE

    barrier()

    for step in range(n_steps):
        # ---- Check revolution flag (ALL threads) ----
        # If revolution already detected, skip work but still hit barriers
        var rev = Int(metrics[moff + 4])

        # ---- Build spatial grid in shared memory: parallel clear + parallel insert ----
        if rev == 0:
            # Clear grid counts and security counts in shared memory
            var clear_i = tid
            while clear_i < GRID_SIZE:
                s_grid_counts[clear_i] = 0
                s_sec_counts[clear_i] = 0
                clear_i += BLOCK_SIZE
        barrier()
        if rev == 0:
            # Insert CITIZENS into grid (security tracked via s_sec_counts only)
            var ins_i = tid
            while ins_i < n_citizens:
                var aj = off + ins_i
                var cell = Int(pos_y[aj]) * GRID_W + Int(pos_x[aj])
                var slot = Int(Atomic.fetch_add(s_grid_counts + cell, Int16(1)))
                if slot < MAX_PER_CELL:
                    s_grid_cells[cell * MAX_PER_CELL + slot] = Int16(ins_i)
                # Load citizen cond
                s_cond[ins_i] = Int8(cond[off + ins_i])
                ins_i += BLOCK_SIZE
            # Insert SECURITY into sec_counts only (not grid_cells)
            ins_i = tid + n_citizens
            while ins_i < n_agents:
                var aj = off + ins_i
                var cell = Int(pos_y[aj]) * GRID_W + Int(pos_x[aj])
                _ = Atomic.fetch_add(s_sec_counts + cell, Int16(1))
                ins_i += BLOCK_SIZE

        barrier()

        # ---- Phase 1: Citizens scan neighbors via spatial grid (PARALLEL) ----
        # Reads from shared memory: s_cond, s_grid_counts (~80x faster)
        if rev == 0:
            var i = tid
            while i < n_citizens:
                var ai = off + i
                if jail_sent[ai] > 0 or s_cond[i] == JAILED_I8:
                    i += BLOCK_SIZE
                    continue

                var ax = Int(pos_x[ai])
                var ay = Int(pos_y[ai])
                var actives: Int = 1
                var opposed: Int = 0
                var support_cnt: Int = 1
                var security: Int = 0

                # Scan cells within vision distance (toroidal wrapping)
                for dy_cell in range(-vision, vision + 1):
                    var cy = (ay + dy_cell + GRID_H) % GRID_H
                    for dx_cell in range(-vision, vision + 1):
                        var cx = (ax + dx_cell + GRID_W) % GRID_W
                        var cell = cy * GRID_W + cx
                        # Security count from pre-computed per-cell counts
                        security += Int(s_sec_counts[cell])
                        # Only iterate citizen agents in this cell
                        var cnt = Int(s_grid_counts[cell])
                        for slot in range(cnt):
                            var j = Int(s_grid_cells[cell * MAX_PER_CELL + slot])
                            if j == i:
                                continue
                            # Grid only contains citizens — no is_citizen check needed
                            var c = s_cond[j]
                            if c == ACTIVE_I8:
                                actives += 1
                            elif c == OPPOSE_I8:
                                opposed += 1
                            elif c == SUPPORT_I8:
                                support_cnt += 1

                var ep = eps_arr[ai]
                var ep_prob = eps_prob_arr[ai]
                var active_ratio = Float32(actives + opposed) / Float32(support_cnt)
                var perception = (Float32(actives) + Float32(opposed) * ep_prob) ** (1.0 / (ep * ep + 1.0))
                var arrest_prob = 1.0 - exp(Float32(-2.3) * Float32(security) / Float32(actives) * 2.0 * ep_prob)
                var opinion = -private_pref[ai] + perception * active_ratio

                var rng_state = rng_arr[ai]
                rng_state = lcg_next(rng_state)
                var rand_act = lcg_float(rng_state)
                rng_arr[ai] = rng_state

                activation_val[ai] = sigmoid_f32(opinion)
                var active_level = sigmoid_f32(opinion - active_th[ai]) - arrest_prob
                var oppose_level = sigmoid_f32(opinion - oppose_th[ai]) - arrest_prob

                if active_level > rand_act:
                    next_cond[ai] = ACTIVE
                elif oppose_level > rand_act:
                    next_cond[ai] = OPPOSE
                else:
                    next_cond[ai] = SUPPORT

                i += BLOCK_SIZE

        barrier()

        # ---- Phase 2: Advance citizens (PARALLEL) ----
        if rev == 0:
            var i = tid
            while i < n_citizens:
                var ai = off + i
                if jail_sent[ai] > 0:
                    jail_sent[ai] -= 1
                    i += BLOCK_SIZE
                    continue
                elif cond[ai] == JAILED:
                    cond[ai] = SUPPORT
                    var rng_state = rng_arr[ai]
                    rng_state = lcg_next(rng_state)
                    pos_x[ai] = Int32(lcg_int(rng_state, GRID_W))
                    rng_state = lcg_next(rng_state)
                    pos_y[ai] = Int32(lcg_int(rng_state, GRID_H))
                    rng_arr[ai] = rng_state

                cond[ai] = next_cond[ai]
                # Move randomly
                var rng_state = rng_arr[ai]
                rng_state = lcg_next(rng_state)
                var choice = lcg_int(rng_state, 9)
                var dx2 = choice // 3 - 1
                var dy2 = choice % 3 - 1
                pos_x[ai] = Int32((Int(pos_x[ai]) + dx2 + GRID_W) % GRID_W)
                pos_y[ai] = Int32((Int(pos_y[ai]) + dy2 + GRID_H) % GRID_H)
                rng_arr[ai] = rng_state

                i += BLOCK_SIZE

        barrier()

        # ---- Rebuild spatial grid (citizens only) after Phase 2 moves ----
        if rev == 0:
            var clear_i = tid
            while clear_i < GRID_SIZE:
                s_grid_counts[clear_i] = 0
                clear_i += BLOCK_SIZE
        barrier()
        if rev == 0:
            var ins_i = tid
            while ins_i < n_citizens:
                var aj = off + ins_i
                var cell = Int(pos_y[aj]) * GRID_W + Int(pos_x[aj])
                var slot = Int(Atomic.fetch_add(s_grid_counts + cell, Int16(1)))
                if slot < MAX_PER_CELL:
                    s_grid_cells[cell * MAX_PER_CELL + slot] = Int16(ins_i)
                ins_i += BLOCK_SIZE

        barrier()

        # ---- Phase 3: Security arrest and move (thread 0 only) ----
        # Sequential to avoid race conditions on arrest targets
        # Uses spatial grid for O(9*MAX_PER_CELL) neighbor lookup instead of O(n_citizens)
        if tid == 0 and rev == 0:
            for s in range(n_citizens, n_agents):
                var ai = off + s
                var sx = Int(pos_x[ai])
                var sy = Int(pos_y[ai])
                var best_active = -1
                var best_oppose = -1

                # Scan Moore neighborhood (3x3) using spatial grid
                # Must pick highest-indexed agent to match linear scan behavior
                for dy_a in range(-1, 2):
                    var cy = (sy + dy_a + GRID_H) % GRID_H
                    for dx_a in range(-1, 2):
                        var cx = (sx + dx_a + GRID_W) % GRID_W
                        var cell = cy * GRID_W + cx
                        var cnt = Int(s_grid_counts[cell])
                        for slot in range(cnt):
                            var j = Int(s_grid_cells[cell * MAX_PER_CELL + slot])
                            # Grid only contains citizens — no type check needed
                            if cond[off + j] == ACTIVE:
                                if j > best_active:
                                    best_active = j
                            elif cond[off + j] == OPPOSE and activation_val[off + j] > threshold_sig:
                                if j > best_oppose:
                                    best_oppose = j

                var arrestee = best_active
                if arrestee < 0:
                    arrestee = best_oppose
                if arrestee >= 0:
                    var arr_idx = off + arrestee
                    var rng_state = rng_arr[ai]
                    rng_state = lcg_next(rng_state)
                    jail_sent[arr_idx] = Int32(lcg_int(rng_state, max_jail))
                    cond[arr_idx] = JAILED
                    rng_arr[ai] = rng_state

                # Move security
                var rng_state2 = rng_arr[ai]
                rng_state2 = lcg_next(rng_state2)
                var choice2 = lcg_int(rng_state2, 9)
                var dx3 = choice2 // 3 - 1
                var dy3 = choice2 % 3 - 1
                pos_x[ai] = Int32((Int(pos_x[ai]) + dx3 + GRID_W) % GRID_W)
                pos_y[ai] = Int32((Int(pos_y[ai]) + dy3 + GRID_H) % GRID_H)
                rng_arr[ai] = rng_state2

        barrier()

        # ---- Revolution check (parallel reduction) ----
        # All threads count active+jailed in their citizen chunk, reduce via shared memory
        if rev == 0:
            var s_reduce = stack_allocation[
                BLOCK_SIZE, Scalar[DType.int32], address_space=AddressSpace.SHARED,
            ]()
            var my_count: Int32 = 0
            var ri = tid
            while ri < n_citizens:
                var c = cond[off + ri]
                if c == ACTIVE or c == JAILED:
                    my_count += 1
                ri += BLOCK_SIZE
            s_reduce[tid] = my_count
            barrier()
            # Tree reduction
            var stride = BLOCK_SIZE // 2
            while stride > 0:
                if tid < stride:
                    s_reduce[tid] = s_reduce[tid] + s_reduce[tid + stride]
                barrier()
                stride //= 2
            if tid == 0:
                if Float32(s_reduce[0]) / Float32(n_citizens) >= 0.95:
                    metrics[moff + 4] = 1

        barrier()

        # ---- Per-step model metrics (thread 0, optional) ----
        # Use `rev` (read at START of step) to decide if work was done.
        # When revolution is detected at step S, rev==0 so work ran and we
        # must count actual state. Steps S+1.. have rev==1, work was skipped,
        # so copy from previous step.
        if tid == 0 and step_metrics:
            var sm_off = sid * MAX_STEPS * N_STEP_FIELDS + step * N_STEP_FIELDS
            if rev == 0:
                # This step was executed -- count actual state
                var s_active: Int32 = 0
                var s_support: Int32 = 0
                var s_oppose: Int32 = 0
                var s_jail: Int32 = 0
                for ci in range(n_citizens):
                    var c = cond[off + ci]
                    if c == ACTIVE:
                        s_active += 1
                    elif c == SUPPORT:
                        s_support += 1
                    elif c == OPPOSE:
                        s_oppose += 1
                    elif c == JAILED:
                        s_jail += 1
                step_metrics[sm_off + 0] = s_active
                step_metrics[sm_off + 1] = s_support
                step_metrics[sm_off + 2] = s_oppose
                step_metrics[sm_off + 3] = s_jail
                step_metrics[sm_off + 4] = Int32(metrics[moff + 4])
            else:
                # Work was skipped -- state is frozen from revolution step
                var prev_off = sm_off - N_STEP_FIELDS
                step_metrics[sm_off + 0] = step_metrics[prev_off + 0]
                step_metrics[sm_off + 1] = step_metrics[prev_off + 1]
                step_metrics[sm_off + 2] = step_metrics[prev_off + 2]
                step_metrics[sm_off + 3] = step_metrics[prev_off + 3]
                step_metrics[sm_off + 4] = Int32(1)

        barrier()

        if trace_cond:
            var trace_step = step + 1
            var tr_i = tid
            while tr_i < n_agents:
                var tr_idx = sid * MAX_TRACE_STEPS * MAX_AGENTS + trace_step * MAX_AGENTS + tr_i
                trace_cond[tr_idx] = cond[off + tr_i]
                trace_pos_x[tr_idx] = pos_x[off + tr_i]
                trace_pos_y[tr_idx] = pos_y[off + tr_i]
                tr_i += BLOCK_SIZE

        barrier()

    # ---- Count final metrics (thread 0) ----
    # metrics[moff + 4] (revolution flag) already set during step loop
    if tid == 0:
        var active_count: Int32 = 0
        var support_count: Int32 = 0
        var oppose_count: Int32 = 0
        var jail_count: Int32 = 0
        for c_i in range(n_citizens):
            var c = cond[off + c_i]
            if c == ACTIVE:
                active_count += 1
            elif c == SUPPORT:
                support_count += 1
            elif c == OPPOSE:
                oppose_count += 1
            elif c == JAILED:
                jail_count += 1

        metrics[moff] = active_count
        metrics[moff + 1] = support_count
        metrics[moff + 2] = oppose_count
        metrics[moff + 3] = jail_count
        # moff + 4 already set by revolution check in step loop
        metrics[moff + 5] = Int32(n_citizens)


# ============================================================
# Host-side initialization and launch
# ============================================================

def main() raises:
    print("=== Cascade Block-per-Sim GPU Kernel ===")

    comptime if not has_accelerator():
        print("No GPU found")
        return

    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    var trace_validation = False
    var args = argv()
    for ai in range(1, len(args)):
        if args[ai] == String("--trace-validation"):
            trace_validation = True

    # Parameter sweep - 5 base seeds x 3 epsilon x 3 sec_density = 45 correctness sims
    # For scale benchmark: 228 seeds x 3 x 3 = 2052 sims
    var benchmark_mode = False
    var seeds = List[Int]()
    seeds.append(42)
    seeds.append(123)
    seeds.append(456)
    seeds.append(789)
    seeds.append(1001)
    if trace_validation:
        seeds = List[Int]()
        seeds.append(16)
    if benchmark_mode:
        for s in range(223):
            seeds.append(2000 + s)

    var epsilons = List[Float32]()
    epsilons.append(0.2)
    epsilons.append(0.5)
    epsilons.append(1.0)
    if trace_validation:
        epsilons = List[Float32]()
        epsilons.append(0.5)

    var sec_densities = List[Float32]()
    sec_densities.append(0.0)
    sec_densities.append(0.02)
    sec_densities.append(0.05)
    if trace_validation:
        sec_densities = List[Float32]()
        sec_densities.append(0.0)

    var num_steps = 50
    var citizen_density = Float32(0.7)
    var pp_mean = Float32(0.0)
    var threshold = Float32(2.94444)
    if trace_validation:
        num_steps = 500
        threshold = Float32(2.5)
    var max_jail = 100
    var vision = 7
    var total = len(seeds) * len(epsilons) * len(sec_densities)

    print("Running", total, "simulations x", num_steps, "steps")
    print("Block size:", BLOCK_SIZE, "threads per sim")
    print("Vision:", vision, "(variable)")
    print()

    # Allocate host buffers (9 params per sim instead of 8)
    var flat_agent_size = total * MAX_AGENTS
    var flat_param_size = total * 9
    var flat_metric_size = total * 6

    var h_params = ctx.enqueue_create_host_buffer[DType.float32](flat_param_size)
    var h_cond = ctx.enqueue_create_host_buffer[DType.int32](flat_agent_size)
    var h_next_cond = ctx.enqueue_create_host_buffer[DType.int32](flat_agent_size)
    var h_pos_x = ctx.enqueue_create_host_buffer[DType.int32](flat_agent_size)
    var h_pos_y = ctx.enqueue_create_host_buffer[DType.int32](flat_agent_size)
    var h_is_citizen = ctx.enqueue_create_host_buffer[DType.int32](flat_agent_size)
    var h_private_pref = ctx.enqueue_create_host_buffer[DType.float32](flat_agent_size)
    var h_eps = ctx.enqueue_create_host_buffer[DType.float32](flat_agent_size)
    var h_eps_prob = ctx.enqueue_create_host_buffer[DType.float32](flat_agent_size)
    var h_oppose_th = ctx.enqueue_create_host_buffer[DType.float32](flat_agent_size)
    var h_active_th = ctx.enqueue_create_host_buffer[DType.float32](flat_agent_size)
    var h_jail_sent = ctx.enqueue_create_host_buffer[DType.int32](flat_agent_size)
    var h_activation_val = ctx.enqueue_create_host_buffer[DType.float32](flat_agent_size)
    var h_rng = ctx.enqueue_create_host_buffer[DType.uint64](flat_agent_size)
    var h_num_citizens = ctx.enqueue_create_host_buffer[DType.int32](total)
    var h_num_agents = ctx.enqueue_create_host_buffer[DType.int32](total)
    var h_metrics = ctx.enqueue_create_host_buffer[DType.int32](flat_metric_size)
    ctx.synchronize()

    # Initialize (identical to cascade_gpu_batch.mojo)
    var sim_idx = 0
    for si in range(len(seeds)):
        for ei in range(len(epsilons)):
            for di in range(len(sec_densities)):
                var seed_val = seeds[si]
                var model_eps = epsilons[ei]
                var sec_dens = sec_densities[di]

                var n_citizens = Int(round(Float64(GRID_SIZE) * Float64(citizen_density)))
                var n_security = Int(round(Float64(GRID_SIZE) * Float64(sec_dens)))
                var n_agents = n_citizens + n_security

                var poff = sim_idx * 9
                h_params[poff + 0] = Float32(seed_val)
                h_params[poff + 1] = Float32(citizen_density)
                h_params[poff + 2] = Float32(sec_dens)
                h_params[poff + 3] = Float32(pp_mean)
                h_params[poff + 4] = Float32(model_eps)
                h_params[poff + 5] = Float32(threshold)
                h_params[poff + 6] = Float32(max_jail)
                h_params[poff + 7] = Float32(num_steps)
                h_params[poff + 8] = Float32(vision)

                h_num_citizens[sim_idx] = Int32(n_citizens)
                h_num_agents[sim_idx] = Int32(n_agents)

                var base_off = sim_idx * MAX_AGENTS

                # Initialize RNG per agent (identical to original)
                var master = UInt64(seed_val)
                for i in range(n_agents):
                    master = lcg_next(master)
                    h_rng[base_off + i] = master ^ UInt64(i * 2654435761)

                # Initialize citizens
                # Matches Python: gauss(pp_mean, std=1), gauss(0, epsilon),
                # gauss(threshold, epsilon) for thresholds
                for i in range(n_citizens):
                    var idx = base_off + i
                    var rng_state = UInt64(h_rng[idx])

                    rng_state = lcg_next(rng_state)
                    h_pos_x[idx] = Int32(lcg_int(rng_state, GRID_W))
                    rng_state = lcg_next(rng_state)
                    h_pos_y[idx] = Int32(lcg_int(rng_state, GRID_H))
                    h_is_citizen[idx] = Int32(1)
                    h_cond[idx] = SUPPORT
                    h_next_cond[idx] = SUPPORT

                    # private_pref ~ gauss(pp_mean, 1.0)
                    var pp_result = lcg_gauss_val(rng_state, pp_mean, Float32(1.0))
                    h_private_pref[idx] = pp_result[0]
                    rng_state = pp_result[1]

                    # epsilon ~ gauss(0, model_eps)
                    var eps_result = lcg_gauss_val(rng_state, Float32(0.0), model_eps)
                    var e = eps_result[0]
                    rng_state = eps_result[1]
                    h_eps[idx] = e
                    h_eps_prob[idx] = sigmoid_f32(e)

                    # thresholds ~ gauss(threshold, epsilon) x2, sorted
                    var t1_result = lcg_gauss_val(rng_state, threshold, e)
                    var t1 = t1_result[0]
                    rng_state = t1_result[1]
                    var t2_result = lcg_gauss_val(rng_state, threshold, e)
                    var t2 = t2_result[0]
                    rng_state = t2_result[1]
                    if t1 < t2:
                        h_oppose_th[idx] = t1
                        h_active_th[idx] = t2
                    else:
                        h_oppose_th[idx] = t2
                        h_active_th[idx] = t1

                    h_jail_sent[idx] = Int32(0)
                    h_activation_val[idx] = Float32(0)
                    h_rng[idx] = rng_state

                # Initialize security (identical to original)
                for i in range(n_citizens, n_agents):
                    var idx = base_off + i
                    var rng_state = UInt64(h_rng[idx])

                    rng_state = lcg_next(rng_state)
                    h_pos_x[idx] = Int32(lcg_int(rng_state, GRID_W))
                    rng_state = lcg_next(rng_state)
                    h_pos_y[idx] = Int32(lcg_int(rng_state, GRID_H))
                    h_is_citizen[idx] = Int32(0)
                    h_cond[idx] = SECURITY_COND
                    h_next_cond[idx] = SECURITY_COND
                    h_jail_sent[idx] = Int32(0)
                    h_activation_val[idx] = Float32(0)
                    h_rng[idx] = rng_state

                sim_idx += 1

    print("Initialized", sim_idx, "simulations on host")

    # Allocate device buffers
    var d_params = ctx.enqueue_create_buffer[DType.float32](flat_param_size)
    var d_cond = ctx.enqueue_create_buffer[DType.int32](flat_agent_size)
    var d_next_cond = ctx.enqueue_create_buffer[DType.int32](flat_agent_size)
    var d_pos_x = ctx.enqueue_create_buffer[DType.int32](flat_agent_size)
    var d_pos_y = ctx.enqueue_create_buffer[DType.int32](flat_agent_size)
    var d_is_citizen = ctx.enqueue_create_buffer[DType.int32](flat_agent_size)
    var d_private_pref = ctx.enqueue_create_buffer[DType.float32](flat_agent_size)
    var d_eps = ctx.enqueue_create_buffer[DType.float32](flat_agent_size)
    var d_eps_prob = ctx.enqueue_create_buffer[DType.float32](flat_agent_size)
    var d_oppose_th = ctx.enqueue_create_buffer[DType.float32](flat_agent_size)
    var d_active_th = ctx.enqueue_create_buffer[DType.float32](flat_agent_size)
    var d_jail_sent = ctx.enqueue_create_buffer[DType.int32](flat_agent_size)
    var d_activation_val = ctx.enqueue_create_buffer[DType.float32](flat_agent_size)
    var d_rng = ctx.enqueue_create_buffer[DType.uint64](flat_agent_size)
    var d_num_citizens = ctx.enqueue_create_buffer[DType.int32](total)
    var d_num_agents = ctx.enqueue_create_buffer[DType.int32](total)
    var d_grid_counts = ctx.enqueue_create_buffer[DType.int32](total * GRID_SIZE)
    var d_grid_cells = ctx.enqueue_create_buffer[DType.int32](total * GRID_SIZE * MAX_PER_CELL)
    var d_metrics = ctx.enqueue_create_buffer[DType.int32](flat_metric_size)

    # Copy host -> device
    ctx.enqueue_copy(dst_buf=d_params, src_buf=h_params)
    ctx.enqueue_copy(dst_buf=d_cond, src_buf=h_cond)
    ctx.enqueue_copy(dst_buf=d_next_cond, src_buf=h_next_cond)
    ctx.enqueue_copy(dst_buf=d_pos_x, src_buf=h_pos_x)
    ctx.enqueue_copy(dst_buf=d_pos_y, src_buf=h_pos_y)
    ctx.enqueue_copy(dst_buf=d_is_citizen, src_buf=h_is_citizen)
    ctx.enqueue_copy(dst_buf=d_private_pref, src_buf=h_private_pref)
    ctx.enqueue_copy(dst_buf=d_eps, src_buf=h_eps)
    ctx.enqueue_copy(dst_buf=d_eps_prob, src_buf=h_eps_prob)
    ctx.enqueue_copy(dst_buf=d_oppose_th, src_buf=h_oppose_th)
    ctx.enqueue_copy(dst_buf=d_active_th, src_buf=h_active_th)
    ctx.enqueue_copy(dst_buf=d_jail_sent, src_buf=h_jail_sent)
    ctx.enqueue_copy(dst_buf=d_activation_val, src_buf=h_activation_val)
    ctx.enqueue_copy(dst_buf=d_rng, src_buf=h_rng)
    ctx.enqueue_copy(dst_buf=d_num_citizens, src_buf=h_num_citizens)
    ctx.enqueue_copy(dst_buf=d_num_agents, src_buf=h_num_agents)
    ctx.synchronize()

    print("Data copied to GPU, launching kernel...")

    # Launch: one BLOCK per simulation
    var t_start = perf_counter_ns()

    # Allocate step_metrics buffer for per-step data collection
    var flat_step_size = total * MAX_STEPS * N_STEP_FIELDS
    var h_step_metrics = ctx.enqueue_create_host_buffer[DType.int32](flat_step_size)
    var d_step_metrics = ctx.enqueue_create_buffer[DType.int32](flat_step_size)
    var flat_trace_size = total * MAX_TRACE_STEPS * MAX_AGENTS
    var h_trace_cond = ctx.enqueue_create_host_buffer[DType.int32](flat_trace_size)
    var h_trace_pos_x = ctx.enqueue_create_host_buffer[DType.int32](flat_trace_size)
    var h_trace_pos_y = ctx.enqueue_create_host_buffer[DType.int32](flat_trace_size)
    var d_trace_cond = ctx.enqueue_create_buffer[DType.int32](flat_trace_size)
    var d_trace_pos_x = ctx.enqueue_create_buffer[DType.int32](flat_trace_size)
    var d_trace_pos_y = ctx.enqueue_create_buffer[DType.int32](flat_trace_size)
    ctx.synchronize()

    ctx.enqueue_function[block_sim_kernel, block_sim_kernel](
        d_params, d_cond, d_next_cond, d_pos_x, d_pos_y,
        d_is_citizen, d_private_pref, d_eps, d_eps_prob,
        d_oppose_th, d_active_th, d_jail_sent, d_activation_val,
        d_rng, d_num_citizens, d_num_agents,
        d_grid_counts, d_grid_cells,
        d_metrics,
        d_step_metrics,
        d_trace_cond, d_trace_pos_x, d_trace_pos_y,
        total,
        grid_dim=total,
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    var elapsed_ns = perf_counter_ns() - t_start
    var elapsed_s = Float64(elapsed_ns) / 1_000_000_000.0

    # Copy metrics back
    ctx.enqueue_copy(dst_buf=h_metrics, src_buf=d_metrics)
    ctx.enqueue_copy(dst_buf=h_step_metrics, src_buf=d_step_metrics)
    ctx.enqueue_copy(dst_buf=h_trace_cond, src_buf=d_trace_cond)
    ctx.enqueue_copy(dst_buf=h_trace_pos_x, src_buf=d_trace_pos_x)
    ctx.enqueue_copy(dst_buf=h_trace_pos_y, src_buf=d_trace_pos_y)
    ctx.synchronize()

    # Print results in same format as cascade_gpu_batch for comparison
    sim_idx = 0
    for si in range(len(seeds)):
        for ei in range(len(epsilons)):
            for di in range(len(sec_densities)):
                var moff = sim_idx * 6
                print(
                    "Sim", sim_idx,
                    "seed=", seeds[si],
                    "eps=", epsilons[ei],
                    "sd=", sec_densities[di],
                    "active=", h_metrics[moff + 0],
                    "support=", h_metrics[moff + 1],
                    "oppose=", h_metrics[moff + 2],
                    "jail=", h_metrics[moff + 3],
                    "rev=", Int32(h_metrics[moff + 4]) > 0,
                )
                # Verify: last step of step_metrics should match final metrics
                var last_step = num_steps - 1
                var sm_off = sim_idx * MAX_STEPS * N_STEP_FIELDS + last_step * N_STEP_FIELDS
                var sm_active = Int32(h_step_metrics[sm_off + 0])
                var sm_support = Int32(h_step_metrics[sm_off + 1])
                var sm_oppose = Int32(h_step_metrics[sm_off + 2])
                var sm_jail = Int32(h_step_metrics[sm_off + 3])
                var final_active = Int32(h_metrics[moff + 0])
                var final_support = Int32(h_metrics[moff + 1])
                var final_oppose = Int32(h_metrics[moff + 2])
                var final_jail = Int32(h_metrics[moff + 3])
                if sm_active != final_active or sm_support != final_support or sm_oppose != final_oppose or sm_jail != final_jail:
                    print("  ** STEP METRICS MISMATCH at sim", sim_idx,
                          "step_metrics=", sm_active, sm_support, sm_oppose, sm_jail,
                          "final=", final_active, final_support, final_oppose, final_jail)
                sim_idx += 1

    if trace_validation:
        print()
        print("=== TRACE CSV ===")
        for sim in range(total):
            var seed_val = seeds[0]
            var eps_val = epsilons[0]
            var sd_val = sec_densities[0]
            var n_citizens = Int(h_num_citizens[sim])
            for step in range(num_steps + 1):
                var tr_base = sim * MAX_TRACE_STEPS * MAX_AGENTS + step * MAX_AGENTS
                for agent_id in range(n_citizens):
                    print(
                        "TRACE,", sim, ",", seed_val, ",", eps_val, ",", sd_val, ",",
                        step, ",", agent_id, ",",
                        h_trace_pos_x[tr_base + agent_id], ",",
                        h_trace_pos_y[tr_base + agent_id], ",",
                        cond_name(h_trace_cond[tr_base + agent_id]),
                        sep="",
                    )

    # Print step-by-step data for first simulation as sample
    print()
    print("=== Per-step data for Sim 0 ===")
    print("step\tactive\tsupport\toppose\tjail\trev")
    for s in range(num_steps):
        var sm_off = 0 * MAX_STEPS * N_STEP_FIELDS + s * N_STEP_FIELDS
        print(s, "\t",
              h_step_metrics[sm_off + 0], "\t",
              h_step_metrics[sm_off + 1], "\t",
              h_step_metrics[sm_off + 2], "\t",
              h_step_metrics[sm_off + 3], "\t",
              h_step_metrics[sm_off + 4])

    print()
    print("Done:", sim_idx, "simulations in", elapsed_s, "seconds (GPU kernel time)")
    print("Throughput:", Float64(sim_idx) / elapsed_s, "sims/sec")
    print("Per-sim:", elapsed_s / Float64(sim_idx), "s")
    print("Architecture: block-per-sim, BLOCK_SIZE =", BLOCK_SIZE)
    print("Vision:", vision, "(variable runtime parameter)")
