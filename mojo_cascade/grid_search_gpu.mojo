"""
Full grid search across model parameters, running batches on GPU.
Processes 2048 simulations per GPU batch for optimal throughput.

Parameter grid:
- Seeds: 30 (1..30)
- pp_mean: 9 values [-1.0 to 1.0]
- security_density: 11 values [0.0 to 0.10]
- epsilon: 9 values [0.05 to 2.0]
- threshold: 7 values [1.5 to 5.0]
Total: 187,110 simulations
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
comptime MAX_AGENTS: Int = 1300  # Must fit citizen_density + max sec_density: 0.7*1600 + 0.1*1600 = 1280
comptime BATCH_SIZE: Int = 2048
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
            pos_x[ai] = Int32((Int(pos_x[ai]) + choice // 3 - 1 + GRID_W) % GRID_W)
            pos_y[ai] = Int32((Int(pos_y[ai]) + choice % 3 - 1 + GRID_H) % GRID_H)
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
            pos_x[ai] = Int32((Int(pos_x[ai]) + choice2 // 3 - 1 + GRID_W) % GRID_W)
            pos_y[ai] = Int32((Int(pos_y[ai]) + choice2 % 3 - 1 + GRID_H) % GRID_H)
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
    metrics[moff] = active_count
    metrics[moff + 1] = support_count
    metrics[moff + 2] = oppose_count
    metrics[moff + 3] = jail_count
    if Float32(active_count + jail_count) / Float32(n_citizens) >= 0.95:
        metrics[moff + 4] = 1
    else:
        metrics[moff + 4] = 0
    metrics[moff + 5] = Int32(n_citizens)






def main() raises:
    print("=== Cascade GPU Grid Search ===")

    comptime if not has_accelerator():
        print("No GPU found")
        return

    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    # Parameter grid
    var seeds = List[Int]()
    for s in range(1, 31):
        seeds.append(s)

    var pp_means = List[Float32]()
    pp_means.append(-1.0)
    pp_means.append(-0.75)
    pp_means.append(-0.5)
    pp_means.append(-0.25)
    pp_means.append(0.0)
    pp_means.append(0.25)
    pp_means.append(0.5)
    pp_means.append(0.75)
    pp_means.append(1.0)

    var sec_densities = List[Float32]()
    sec_densities.append(0.0)
    sec_densities.append(0.005)
    sec_densities.append(0.01)
    sec_densities.append(0.015)
    sec_densities.append(0.02)
    sec_densities.append(0.025)
    sec_densities.append(0.03)
    sec_densities.append(0.04)
    sec_densities.append(0.05)
    sec_densities.append(0.07)
    sec_densities.append(0.10)

    var epsilons = List[Float32]()
    epsilons.append(0.05)
    epsilons.append(0.1)
    epsilons.append(0.2)
    epsilons.append(0.3)
    epsilons.append(0.5)
    epsilons.append(0.7)
    epsilons.append(1.0)
    epsilons.append(1.5)
    epsilons.append(2.0)

    var thresholds = List[Float32]()
    thresholds.append(1.5)
    thresholds.append(2.0)
    thresholds.append(2.5)
    thresholds.append(2.94444)
    thresholds.append(3.5)
    thresholds.append(4.0)
    thresholds.append(5.0)

    var citizen_density = Float32(0.7)
    var total_sims = len(seeds) * len(pp_means) * len(sec_densities) * len(epsilons) * len(thresholds)
    var num_batches = (total_sims + BATCH_SIZE - 1) // BATCH_SIZE

    print("Grid: seeds=", len(seeds), "pp_means=", len(pp_means),
          "sec_dens=", len(sec_densities), "eps=", len(epsilons),
          "thresholds=", len(thresholds))
    print("Total simulations:", total_sims)
    print("Batch size:", BATCH_SIZE, "Num batches:", num_batches)
    print("Estimated time:", Float64(total_sims) / 38.7, "seconds")
    print()

    # Allocate buffers
    comptime flat_agent_size: Int = BATCH_SIZE * MAX_AGENTS
    comptime flat_param_size: Int = BATCH_SIZE * 8
    comptime flat_metric_size: Int = BATCH_SIZE * 6

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
    var h_num_citizens = ctx.enqueue_create_host_buffer[DType.int32](BATCH_SIZE)
    var h_num_agents = ctx.enqueue_create_host_buffer[DType.int32](BATCH_SIZE)
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
    var d_num_citizens = ctx.enqueue_create_buffer[DType.int32](BATCH_SIZE)
    var d_num_agents = ctx.enqueue_create_buffer[DType.int32](BATCH_SIZE)
    var d_metrics = ctx.enqueue_create_buffer[DType.int32](flat_metric_size)
    ctx.synchronize()

    # Output CSV header
    print("seed,pp_mean,sec_density,epsilon,threshold,active,support,oppose,jail,revolution")

    var t_total_start = perf_counter_ns()
    var total_processed = 0
    var batch_idx = 0

    # Dimension sizes for index computation
    var n_seeds = len(seeds)
    var n_pp = len(pp_means)
    var n_sd = len(sec_densities)
    var n_eps = len(epsilons)
    var n_th = len(thresholds)

    var combo_idx = 0
    while combo_idx < total_sims:
        var n_this_batch = min(BATCH_SIZE, total_sims - combo_idx)

        # Initialize batch on host - compute params from flat index
        for b in range(n_this_batch):
            var ci = combo_idx + b
            # Decode flat index: seeds x pp_means x sec_dens x epsilons x thresholds
            var ti_idx = ci % n_th
            var rem = ci // n_th
            var ei_idx = rem % n_eps
            rem = rem // n_eps
            var di_idx = rem % n_sd
            rem = rem // n_sd
            var pi_idx = rem % n_pp
            var si_idx = rem // n_pp

            var seed_val = seeds[si_idx]
            var model_eps = epsilons[ei_idx]
            var sec_dens = sec_densities[di_idx]
            var pp_mean_val = pp_means[pi_idx]
            var threshold_val = thresholds[ti_idx]

            var n_citizens = Int(round(Float64(GRID_SIZE) * Float64(citizen_density)))
            var n_security = Int(round(Float64(GRID_SIZE) * Float64(sec_dens)))
            var n_agents = n_citizens + n_security

            var poff = b * 8
            h_params[poff] = Float32(seed_val)
            h_params[poff + 1] = citizen_density
            h_params[poff + 2] = sec_dens
            h_params[poff + 3] = pp_mean_val
            h_params[poff + 4] = model_eps
            h_params[poff + 5] = threshold_val
            h_params[poff + 6] = Float32(100)
            h_params[poff + 7] = Float32(NUM_STEPS)

            h_num_citizens[b] = Int32(n_citizens)
            h_num_agents[b] = Int32(n_agents)

            var base_off = b * MAX_AGENTS
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
                h_private_pref[idx] = pp_mean_val + (lcg_float(rng_state) * 2.0 - 1.0)
                rng_state = lcg_next(rng_state)
                var e = model_eps * (lcg_float(rng_state) * 2.0 - 1.0)
                h_eps[idx] = e
                h_eps_prob[idx] = sigmoid_f32(e)
                rng_state = lcg_next(rng_state)
                var t1 = threshold_val + e * (lcg_float(rng_state) * 2.0 - 1.0)
                rng_state = lcg_next(rng_state)
                var t2 = threshold_val + e * (lcg_float(rng_state) * 2.0 - 1.0)
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

        # Copy to GPU
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
        var block_size = min(64, n_this_batch)
        var grid_size = (n_this_batch + block_size - 1) // block_size

        ctx.enqueue_function[sim_kernel, sim_kernel](
            d_params, d_cond, d_next_cond, d_pos_x, d_pos_y,
            d_is_citizen, d_private_pref, d_eps, d_eps_prob,
            d_oppose_th, d_active_th, d_jail_sent, d_activation_val,
            d_rng, d_num_citizens, d_num_agents, d_metrics,
            n_this_batch,
            grid_dim=grid_size,
            block_dim=block_size,
        )

        # Copy metrics back
        ctx.enqueue_copy(dst_buf=h_metrics, src_buf=d_metrics)
        ctx.synchronize()

        # Output results
        for b in range(n_this_batch):
            var ci = combo_idx + b
            # Recompute params from flat index for output
            var ti_idx = ci % n_th
            var rem2 = ci // n_th
            var ei_idx = rem2 % n_eps
            rem2 = rem2 // n_eps
            var di_idx = rem2 % n_sd
            rem2 = rem2 // n_sd
            var pi_idx = rem2 % n_pp
            var si_idx = rem2 // n_pp

            var moff = b * 6
            print(
                seeds[si_idx], ",",
                pp_means[pi_idx], ",",
                sec_densities[di_idx], ",",
                epsilons[ei_idx], ",",
                thresholds[ti_idx], ",",
                h_metrics[moff], ",",
                h_metrics[moff + 1], ",",
                h_metrics[moff + 2], ",",
                h_metrics[moff + 3], ",",
                h_metrics[moff + 4],
            )

        total_processed += n_this_batch
        combo_idx += n_this_batch
        batch_idx += 1

        # Progress (to stderr via print to not pollute CSV)
        if batch_idx % 10 == 0 or combo_idx >= total_sims:
            var elapsed_s = Float64(perf_counter_ns() - t_total_start) / 1_000_000_000.0
            var remaining = Float64(total_sims - total_processed) / (Float64(total_processed) / elapsed_s)
            # Use print since we can't write to stderr easily
            # Results are parseable because they have commas

    var total_elapsed_s = Float64(perf_counter_ns() - t_total_start) / 1_000_000_000.0
    print()
    print("# SUMMARY")
    print("# Total:", total_processed, "simulations in", total_elapsed_s, "seconds")
    print("# Throughput:", Float64(total_processed) / total_elapsed_s, "sims/sec")
    print("# Batches:", batch_idx, "x", BATCH_SIZE, "sims")
