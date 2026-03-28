"""
Resistance Cascade ABM - GPU Batch Simulation.

Runs all simulations in parallel on GPU. Each GPU thread runs one complete
simulation independently. Agent data stored in flat arrays indexed by
(sim_id * MAX_AGENTS + agent_id) for GPU-friendly memory access.

Uses UnsafePointer kernel parameters for direct scalar access.
"""

from std.sys import has_accelerator
from std.math import exp
from std.collections import List
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx, block_dim, global_idx


# ============================================================
# Constants
# ============================================================
comptime SUPPORT: Int32 = 0
comptime ACTIVE: Int32 = 1
comptime OPPOSE: Int32 = 2
comptime JAILED: Int32 = 3
comptime SECURITY_COND: Int32 = 4

comptime GRID_W: Int = 40
comptime GRID_H: Int = 40
comptime GRID_SIZE: Int = GRID_W * GRID_H
comptime VISION: Int = 7

comptime MAX_AGENTS: Int = 1200

# Batch dimensions
comptime NUM_SIMS: Int = 45


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


@always_inline
fn sigmoid_f32(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


# ============================================================
# GPU Kernel: Run one simulation per thread
# ============================================================

def sim_kernel(
    # Per-sim params: [sim_id * 8 + param_idx]
    # 0=seed, 1=cit_dens, 2=sec_dens, 3=pp_mean, 4=eps, 5=threshold, 6=max_jail, 7=n_steps
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
    # Output metrics: [sim_id * 6 + metric_idx]
    metrics: UnsafePointer[Int32, MutAnyOrigin],
):
    var sid = Int(global_idx.x)
    if sid >= NUM_SIMS:
        return

    var off = sid * MAX_AGENTS
    var poff = sid * 8

    var n_citizens = Int(num_citizens_arr[sid])
    var n_agents = Int(num_agents_arr[sid])
    var threshold_val = params[poff + 5]
    var threshold_sig = sigmoid_f32(threshold_val)
    var max_jail = Int(params[poff + 6])
    var n_steps = Int(params[poff + 7])

    for step in range(n_steps):
        # Phase 1: Citizens scan neighbors
        for i in range(n_citizens):
            var ai = off + i
            if jail_sent[ai] > 0 or cond[ai] == JAILED:
                continue

            var ax = Int(pos_x[ai])
            var ay = Int(pos_y[ai])
            var actives: Int = 1
            var opposed: Int = 0
            var support_cnt: Int = 1
            var security: Int = 0

            for j in range(n_agents):
                if j == i:
                    continue
                var aj = off + j
                var dx = abs(Int(pos_x[aj]) - ax)
                var dy = abs(Int(pos_y[aj]) - ay)
                if dx > GRID_W // 2:
                    dx = GRID_W - dx
                if dy > GRID_H // 2:
                    dy = GRID_H - dy
                if dx <= VISION and dy <= VISION:
                    if is_citizen[aj] == 1:
                        var c = cond[aj]
                        if c == ACTIVE:
                            actives += 1
                        elif c == OPPOSE:
                            opposed += 1
                        elif c == SUPPORT:
                            support_cnt += 1
                    else:
                        security += 1

            var active_ratio = Float32(actives + opposed) / Float32(support_cnt)
            var ep = eps_arr[ai]
            var ep_prob = eps_prob_arr[ai]
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

        # Phase 2: Advance citizens
        for i in range(n_citizens):
            var ai = off + i
            if jail_sent[ai] > 0:
                jail_sent[ai] -= 1
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
            # Move
            var rng_state = rng_arr[ai]
            rng_state = lcg_next(rng_state)
            var choice = lcg_int(rng_state, 9)
            var dx2 = choice // 3 - 1
            var dy2 = choice % 3 - 1
            pos_x[ai] = Int32((Int(pos_x[ai]) + dx2 + GRID_W) % GRID_W)
            pos_y[ai] = Int32((Int(pos_y[ai]) + dy2 + GRID_H) % GRID_H)
            rng_arr[ai] = rng_state

        # Phase 3: Security arrest and move
        for i in range(n_citizens, n_agents):
            var ai = off + i
            var sx = Int(pos_x[ai])
            var sy = Int(pos_y[ai])
            var best_active = -1
            var best_oppose = -1

            for j in range(n_citizens):
                var aj = off + j
                var dx = abs(Int(pos_x[aj]) - sx)
                var dy = abs(Int(pos_y[aj]) - sy)
                if dx > GRID_W // 2:
                    dx = GRID_W - dx
                if dy > GRID_H // 2:
                    dy = GRID_H - dy
                if dx <= 1 and dy <= 1:
                    if cond[aj] == ACTIVE:
                        best_active = j
                    elif cond[aj] == OPPOSE and activation_val[aj] > threshold_sig:
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

    # Count final metrics
    var moff = sid * 6
    var active_count: Int32 = 0
    var support_count: Int32 = 0
    var oppose_count: Int32 = 0
    var jail_count: Int32 = 0
    for i in range(n_citizens):
        var c = cond[off + i]
        if c == ACTIVE:
            active_count += 1
        elif c == SUPPORT:
            support_count += 1
        elif c == OPPOSE:
            oppose_count += 1
        elif c == JAILED:
            jail_count += 1

    metrics[moff + 0] = active_count
    metrics[moff + 1] = support_count
    metrics[moff + 2] = oppose_count
    metrics[moff + 3] = jail_count
    var tot = active_count + jail_count
    if Float32(tot) / Float32(n_citizens) >= 0.95:
        metrics[moff + 4] = 1
    else:
        metrics[moff + 4] = 0
    metrics[moff + 5] = Int32(n_citizens)


# ============================================================
# Host-side initialization and launch
# ============================================================

def main() raises:
    print("=== Cascade GPU Batch Simulation (Mojo) ===")
    print("GPU available:", has_accelerator())

    comptime if not has_accelerator():
        print("ERROR: No GPU accelerator found.")
        return

    # Parameter sweep
    var seeds = List[Int]()
    seeds.append(42)
    seeds.append(123)
    seeds.append(456)
    seeds.append(789)
    seeds.append(1001)

    var epsilons = List[Float32]()
    epsilons.append(0.2)
    epsilons.append(0.5)
    epsilons.append(1.0)

    var sec_densities = List[Float32]()
    sec_densities.append(0.0)
    sec_densities.append(0.02)
    sec_densities.append(0.05)

    var num_steps = 50
    var citizen_density = Float32(0.7)
    var pp_mean = Float32(0.0)
    var threshold = Float32(2.94444)
    var max_jail = 100
    var total = len(seeds) * len(epsilons) * len(sec_densities)

    print("Running", total, "simulations x", num_steps, "steps ON GPU")
    print()

    var ctx = DeviceContext()

    # Allocate host buffers
    comptime flat_agent_size: Int = NUM_SIMS * MAX_AGENTS
    comptime flat_param_size: Int = NUM_SIMS * 8
    comptime flat_metric_size: Int = NUM_SIMS * 6

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
    var h_num_citizens = ctx.enqueue_create_host_buffer[DType.int32](NUM_SIMS)
    var h_num_agents = ctx.enqueue_create_host_buffer[DType.int32](NUM_SIMS)
    var h_metrics = ctx.enqueue_create_host_buffer[DType.int32](flat_metric_size)
    ctx.synchronize()

    # Initialize all agent data per simulation
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

                var poff = sim_idx * 8
                h_params[poff + 0] = Float32(seed_val)
                h_params[poff + 1] = Float32(citizen_density)
                h_params[poff + 2] = Float32(sec_dens)
                h_params[poff + 3] = Float32(pp_mean)
                h_params[poff + 4] = Float32(model_eps)
                h_params[poff + 5] = Float32(threshold)
                h_params[poff + 6] = Float32(max_jail)
                h_params[poff + 7] = Float32(num_steps)

                h_num_citizens[sim_idx] = Int32(n_citizens)
                h_num_agents[sim_idx] = Int32(n_agents)

                var base_off = sim_idx * MAX_AGENTS

                # Initialize RNG per agent
                var master = UInt64(seed_val)
                for i in range(n_agents):
                    master = lcg_next(master)
                    h_rng[base_off + i] = master ^ UInt64(i * 2654435761)

                # Initialize citizens
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

                    rng_state = lcg_next(rng_state)
                    h_private_pref[idx] = pp_mean + Float32(1.0) * (lcg_float(rng_state) * 2.0 - 1.0)

                    rng_state = lcg_next(rng_state)
                    var e = model_eps * (lcg_float(rng_state) * 2.0 - 1.0)
                    h_eps[idx] = e
                    h_eps_prob[idx] = sigmoid_f32(e)

                    rng_state = lcg_next(rng_state)
                    var t1 = threshold + e * (lcg_float(rng_state) * 2.0 - 1.0)
                    rng_state = lcg_next(rng_state)
                    var t2 = threshold + e * (lcg_float(rng_state) * 2.0 - 1.0)
                    if t1 < t2:
                        h_oppose_th[idx] = t1
                        h_active_th[idx] = t2
                    else:
                        h_oppose_th[idx] = t2
                        h_active_th[idx] = t1

                    h_jail_sent[idx] = Int32(0)
                    h_activation_val[idx] = Float32(0)
                    h_rng[idx] = rng_state

                # Initialize security
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
    var d_num_citizens = ctx.enqueue_create_buffer[DType.int32](NUM_SIMS)
    var d_num_agents = ctx.enqueue_create_buffer[DType.int32](NUM_SIMS)
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

    # Launch: one thread per simulation (all run in parallel)
    var t_start = perf_counter_ns()

    ctx.enqueue_function[sim_kernel, sim_kernel](
        d_params, d_cond, d_next_cond, d_pos_x, d_pos_y,
        d_is_citizen, d_private_pref, d_eps, d_eps_prob,
        d_oppose_th, d_active_th, d_jail_sent, d_activation_val,
        d_rng, d_num_citizens, d_num_agents, d_metrics,
        grid_dim=1,
        block_dim=NUM_SIMS,
    )
    ctx.synchronize()

    var elapsed_ns = perf_counter_ns() - t_start
    var elapsed_s = Float64(elapsed_ns) / 1_000_000_000.0

    # Copy metrics back
    ctx.enqueue_copy(dst_buf=h_metrics, src_buf=d_metrics)
    ctx.synchronize()

    # Print results
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
                sim_idx += 1

    print()
    print("Done:", sim_idx, "simulations in", elapsed_s, "seconds (GPU kernel time)")
    print("Throughput:", Float64(sim_idx) / elapsed_s, "sims/sec")
    print("Per-sim:", elapsed_s / Float64(sim_idx), "s")
    print()
    print("Comparison: CPU Mojo sequential = ~20s for 45 sims")
    print("GPU speedup over CPU Mojo =", 20.0 / elapsed_s, "x")
    print("GPU speedup over Python =", 75.9 / elapsed_s, "x")
