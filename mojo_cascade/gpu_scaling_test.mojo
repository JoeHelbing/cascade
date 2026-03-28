"""
GPU scaling test: measure throughput at different batch sizes to find
practical parallelism limits on the local GPU.
"""

from std.sys import has_accelerator
from std.math import exp
from std.collections import List
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx, block_dim, global_idx


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

# Test with increasing batch sizes
comptime MAX_BATCH: Int = 2048
comptime NUM_STEPS: Int = 50


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


def sim_kernel(
    params: UnsafePointer[Float32, MutAnyOrigin],
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
    num_citizens_arr: UnsafePointer[Int32, MutAnyOrigin],
    num_agents_arr: UnsafePointer[Int32, MutAnyOrigin],
    metrics: UnsafePointer[Int32, MutAnyOrigin],
    num_sims: Int,
):
    var sid = Int(global_idx.x)
    if sid >= num_sims:
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
            var rng_state = rng_arr[ai]
            rng_state = lcg_next(rng_state)
            var choice = lcg_int(rng_state, 9)
            var dx2 = choice // 3 - 1
            var dy2 = choice % 3 - 1
            pos_x[ai] = Int32((Int(pos_x[ai]) + dx2 + GRID_W) % GRID_W)
            pos_y[ai] = Int32((Int(pos_y[ai]) + dy2 + GRID_H) % GRID_H)
            rng_arr[ai] = rng_state

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

            var rng_state2 = rng_arr[ai]
            rng_state2 = lcg_next(rng_state2)
            var choice2 = lcg_int(rng_state2, 9)
            var dx3 = choice2 // 3 - 1
            var dy3 = choice2 % 3 - 1
            pos_x[ai] = Int32((Int(pos_x[ai]) + dx3 + GRID_W) % GRID_W)
            pos_y[ai] = Int32((Int(pos_y[ai]) + dy3 + GRID_H) % GRID_H)
            rng_arr[ai] = rng_state2

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


def main() raises:
    print("=== GPU Scaling Test: RTX 3090 ===")

    comptime if not has_accelerator():
        print("No GPU found")
        return

    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    print()

    # Test batch sizes: 1, 4, 16, 45, 64, 128, 256, 512, 1024, 2048
    var batch_sizes = List[Int]()
    batch_sizes.append(1)
    batch_sizes.append(4)
    batch_sizes.append(16)
    batch_sizes.append(45)
    batch_sizes.append(64)
    batch_sizes.append(128)
    batch_sizes.append(256)
    batch_sizes.append(512)
    batch_sizes.append(1024)
    batch_sizes.append(2048)

    var citizen_density = Float32(0.7)
    var pp_mean = Float32(0.0)
    var threshold = Float32(2.94444)
    var max_jail = 100
    var num_steps = 50

    comptime flat_agent_size: Int = MAX_BATCH * MAX_AGENTS
    comptime flat_param_size: Int = MAX_BATCH * 8
    comptime flat_metric_size: Int = MAX_BATCH * 6

    # Allocate max-size buffers once
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
    var h_num_citizens = ctx.enqueue_create_host_buffer[DType.int32](MAX_BATCH)
    var h_num_agents = ctx.enqueue_create_host_buffer[DType.int32](MAX_BATCH)
    var h_metrics = ctx.enqueue_create_host_buffer[DType.int32](flat_metric_size)

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
    var d_num_citizens = ctx.enqueue_create_buffer[DType.int32](MAX_BATCH)
    var d_num_agents = ctx.enqueue_create_buffer[DType.int32](MAX_BATCH)
    var d_metrics = ctx.enqueue_create_buffer[DType.int32](flat_metric_size)
    ctx.synchronize()

    print("batch_size\tkernel_time_s\tsims_per_sec\ttime_per_sim_ms\tspeedup_vs_1")

    var baseline_per_sim = Float64(0)

    for bi in range(len(batch_sizes)):
        var n_sims = batch_sizes[bi]

        # Initialize all sims with varying parameters
        for sim_idx in range(n_sims):
            var seed_val = 42 + sim_idx * 7
            var model_eps = Float32(0.2) + Float32(sim_idx % 10) * Float32(0.1)
            var sec_dens = Float32(sim_idx % 3) * Float32(0.02)

            var n_citizens = Int(round(Float64(GRID_SIZE) * Float64(citizen_density)))
            var n_security = Int(round(Float64(GRID_SIZE) * Float64(sec_dens)))
            var n_agents = n_citizens + n_security

            var poff = sim_idx * 8
            h_params[poff + 0] = Float32(seed_val)
            h_params[poff + 1] = citizen_density
            h_params[poff + 2] = sec_dens
            h_params[poff + 3] = pp_mean
            h_params[poff + 4] = model_eps
            h_params[poff + 5] = threshold
            h_params[poff + 6] = Float32(max_jail)
            h_params[poff + 7] = Float32(num_steps)

            h_num_citizens[sim_idx] = Int32(n_citizens)
            h_num_agents[sim_idx] = Int32(n_agents)

            var base_off = sim_idx * MAX_AGENTS
            var master = UInt64(seed_val)
            for i in range(n_agents):
                master = lcg_next(master)
                h_rng[base_off + i] = master ^ UInt64(i * 2654435761)

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
                h_private_pref[idx] = pp_mean + (lcg_float(rng_state) * 2.0 - 1.0)
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

        # Copy to device
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

        # Launch kernel
        var block_size = 64
        if n_sims <= 64:
            block_size = n_sims
        var grid_size = (n_sims + block_size - 1) // block_size

        var t_start = perf_counter_ns()
        ctx.enqueue_function[sim_kernel, sim_kernel](
            d_params, d_cond, d_next_cond, d_pos_x, d_pos_y,
            d_is_citizen, d_private_pref, d_eps, d_eps_prob,
            d_oppose_th, d_active_th, d_jail_sent, d_activation_val,
            d_rng, d_num_citizens, d_num_agents, d_metrics,
            n_sims,
            grid_dim=grid_size,
            block_dim=block_size,
        )
        ctx.synchronize()
        var elapsed_ns = perf_counter_ns() - t_start
        var elapsed_s = Float64(elapsed_ns) / 1_000_000_000.0
        var sims_per_sec = Float64(n_sims) / elapsed_s
        var time_per_sim_ms = elapsed_s / Float64(n_sims) * 1000.0

        if bi == 0:
            baseline_per_sim = elapsed_s / Float64(n_sims)

        var speedup = baseline_per_sim / (elapsed_s / Float64(n_sims))

        print(
            n_sims, "\t",
            elapsed_s, "\t",
            sims_per_sec, "\t",
            time_per_sim_ms, "\t",
            speedup,
        )

    print()
    print("RTX 3090: 82 SMs, 10496 CUDA cores")
    print("Each simulation: ~1120 agents, 50 steps, O(n^2) neighbor scan")
