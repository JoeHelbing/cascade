"""
Resistance Cascade ABM - Mojo-compiled batch simulation.

Runs multiple simulation instances. Each simulation has agents stored as
struct-of-arrays for cache-friendly access. Compiled Mojo gives significant
speedup over Python even on CPU.
"""

from std.sys import has_accelerator
from std.math import exp
from std.collections import List
from std.time import perf_counter_ns


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
# Simulation State
# ============================================================

struct SimState:
    var num_agents: Int
    var num_citizens: Int
    var num_security: Int

    # Per-agent arrays (struct of arrays)
    var cond: List[Int32]
    var next_cond: List[Int32]
    var pos_x: List[Int]
    var pos_y: List[Int]
    var is_citizen_flag: List[Bool]

    var private_pref: List[Float32]
    var eps: List[Float32]
    var eps_prob: List[Float32]
    var oppose_th: List[Float32]
    var active_th: List[Float32]
    var jail_sent: List[Int]
    var did_flip: List[Bool]
    var ever_flip: List[Bool]
    var activation_val: List[Float32]

    var rng: List[UInt64]

    # Model params
    var seed_val: Int
    var threshold: Float32
    var threshold_sig: Float32
    var max_jail: Int
    var sec_density: Float32

    # Metrics
    var active_count: Int
    var support_count: Int
    var oppose_count: Int
    var jail_count: Int
    var revolution: Bool
    var flip_count: Int

    fn __init__(out self, seed_val: Int, citizen_density: Float32,
                sec_density: Float32, pp_mean: Float32,
                model_eps: Float32, threshold: Float32):
        self.seed_val = seed_val
        self.sec_density = sec_density
        self.threshold = threshold
        self.threshold_sig = sigmoid_f32(threshold)
        self.max_jail = 100

        self.num_citizens = Int(round(Float64(GRID_SIZE) * Float64(citizen_density)))
        self.num_security = Int(round(Float64(GRID_SIZE) * Float64(sec_density)))
        self.num_agents = self.num_citizens + self.num_security

        var n = self.num_agents
        self.cond = List[Int32](length=n, fill=SUPPORT)
        self.next_cond = List[Int32](length=n, fill=SUPPORT)
        self.pos_x = List[Int](length=n, fill=0)
        self.pos_y = List[Int](length=n, fill=0)
        self.is_citizen_flag = List[Bool](length=n, fill=True)
        self.private_pref = List[Float32](length=n, fill=Float32(0))
        self.eps = List[Float32](length=n, fill=Float32(0))
        self.eps_prob = List[Float32](length=n, fill=Float32(0))
        self.oppose_th = List[Float32](length=n, fill=Float32(0))
        self.active_th = List[Float32](length=n, fill=Float32(0))
        self.jail_sent = List[Int](length=n, fill=0)
        self.did_flip = List[Bool](length=n, fill=False)
        self.ever_flip = List[Bool](length=n, fill=False)
        self.activation_val = List[Float32](length=n, fill=Float32(0))
        self.rng = List[UInt64](length=n, fill=UInt64(0))

        self.active_count = 0
        self.support_count = self.num_citizens
        self.oppose_count = 0
        self.jail_count = 0
        self.revolution = False
        self.flip_count = 0

        # Seed RNG per agent
        var master = UInt64(seed_val)
        for i in range(n):
            master = lcg_next(master)
            self.rng[i] = master ^ UInt64(i * 2654435761)

        # Place citizens
        for i in range(self.num_citizens):
            self.rng[i] = lcg_next(self.rng[i])
            self.pos_x[i] = lcg_int(self.rng[i], GRID_W)
            self.rng[i] = lcg_next(self.rng[i])
            self.pos_y[i] = lcg_int(self.rng[i], GRID_H)
            self.is_citizen_flag[i] = True

            # Citizen parameters
            self.rng[i] = lcg_next(self.rng[i])
            self.private_pref[i] = pp_mean + Float32(1.0) * (lcg_float(self.rng[i]) * 2.0 - 1.0)

            self.rng[i] = lcg_next(self.rng[i])
            var e = model_eps * (lcg_float(self.rng[i]) * 2.0 - 1.0)
            self.eps[i] = e
            self.eps_prob[i] = sigmoid_f32(e)

            self.rng[i] = lcg_next(self.rng[i])
            var t1 = threshold + e * (lcg_float(self.rng[i]) * 2.0 - 1.0)
            self.rng[i] = lcg_next(self.rng[i])
            var t2 = threshold + e * (lcg_float(self.rng[i]) * 2.0 - 1.0)
            if t1 < t2:
                self.oppose_th[i] = t1
                self.active_th[i] = t2
            else:
                self.oppose_th[i] = t2
                self.active_th[i] = t1

        # Place security
        for i in range(self.num_citizens, self.num_agents):
            self.rng[i] = lcg_next(self.rng[i])
            self.pos_x[i] = lcg_int(self.rng[i], GRID_W)
            self.rng[i] = lcg_next(self.rng[i])
            self.pos_y[i] = lcg_int(self.rng[i], GRID_H)
            self.is_citizen_flag[i] = False
            self.cond[i] = SECURITY_COND


    fn step(mut self):
        """Run one simulation step."""
        # Phase 1: Citizens scan neighbors
        for i in range(self.num_citizens):
            self.did_flip[i] = False
            if self.jail_sent[i] > 0 or self.cond[i] == JAILED:
                continue

            var ax = self.pos_x[i]
            var ay = self.pos_y[i]
            var actives: Int = 1
            var opposed: Int = 0
            var support: Int = 1
            var security: Int = 0

            # Scan all agents to find neighbors in vision
            for j in range(self.num_agents):
                if j == i:
                    continue
                var dx = abs(self.pos_x[j] - ax)
                var dy = abs(self.pos_y[j] - ay)
                if dx > GRID_W // 2:
                    dx = GRID_W - dx
                if dy > GRID_H // 2:
                    dy = GRID_H - dy
                if dx <= VISION and dy <= VISION:
                    if self.is_citizen_flag[j]:
                        var c = self.cond[j]
                        if c == ACTIVE:
                            actives += 1
                        elif c == OPPOSE:
                            opposed += 1
                        elif c == SUPPORT:
                            support += 1
                    else:
                        security += 1

            # Determine condition
            var active_ratio = Float32(actives + opposed) / Float32(support)
            var ep = self.eps[i]
            var perception = (Float32(actives) + Float32(opposed) * self.eps_prob[i]) ** (1.0 / (ep * ep + 1.0))
            var arrest_prob = 1.0 - exp(Float32(-2.3) * Float32(security) / Float32(actives) * 2.0 * self.eps_prob[i])
            var opinion = -self.private_pref[i] + perception * active_ratio

            self.rng[i] = lcg_next(self.rng[i])
            var rand_act = lcg_float(self.rng[i])

            self.activation_val[i] = sigmoid_f32(opinion)
            var active_level = sigmoid_f32(opinion - self.active_th[i]) - arrest_prob
            var oppose_level = sigmoid_f32(opinion - self.oppose_th[i]) - arrest_prob

            if active_level > rand_act:
                if self.next_cond[i] != ACTIVE:
                    self.did_flip[i] = True
                    self.ever_flip[i] = True
                self.next_cond[i] = ACTIVE
            elif oppose_level > rand_act:
                self.next_cond[i] = OPPOSE
            else:
                self.next_cond[i] = SUPPORT

        # Phase 2: Advance citizens
        for i in range(self.num_citizens):
            if self.jail_sent[i] > 0:
                self.jail_sent[i] -= 1
                continue
            elif self.cond[i] == JAILED:
                self.cond[i] = SUPPORT
                self.rng[i] = lcg_next(self.rng[i])
                self.pos_x[i] = lcg_int(self.rng[i], GRID_W)
                self.rng[i] = lcg_next(self.rng[i])
                self.pos_y[i] = lcg_int(self.rng[i], GRID_H)

            self.cond[i] = self.next_cond[i]
            self._move(i)

        # Phase 3: Security arrest and move
        for i in range(self.num_citizens, self.num_agents):
            self._arrest(i)
            self._move(i)

        # Count
        self.active_count = 0
        self.support_count = 0
        self.oppose_count = 0
        self.jail_count = 0
        self.flip_count = 0
        for i in range(self.num_citizens):
            var c = self.cond[i]
            if c == ACTIVE:
                self.active_count += 1
            elif c == SUPPORT:
                self.support_count += 1
            elif c == OPPOSE:
                self.oppose_count += 1
            elif c == JAILED:
                self.jail_count += 1
            if self.did_flip[i]:
                self.flip_count += 1

        var tot = self.active_count + self.jail_count
        if Float32(tot) / Float32(self.num_citizens) >= 0.95:
            self.revolution = True


    fn _move(mut self, idx: Int):
        var ax = self.pos_x[idx]
        var ay = self.pos_y[idx]
        self.rng[idx] = lcg_next(self.rng[idx])
        var choice = lcg_int(self.rng[idx], 9)
        var dx = choice // 3 - 1
        var dy = choice % 3 - 1
        self.pos_x[idx] = (ax + dx + GRID_W) % GRID_W
        self.pos_y[idx] = (ay + dy + GRID_H) % GRID_H


    fn _arrest(mut self, sec_idx: Int):
        var sx = self.pos_x[sec_idx]
        var sy = self.pos_y[sec_idx]
        var best_active = -1
        var best_oppose = -1

        for j in range(self.num_citizens):
            var dx = abs(self.pos_x[j] - sx)
            var dy = abs(self.pos_y[j] - sy)
            if dx > GRID_W // 2:
                dx = GRID_W - dx
            if dy > GRID_H // 2:
                dy = GRID_H - dy
            if dx <= 1 and dy <= 1:
                if self.cond[j] == ACTIVE:
                    best_active = j
                elif self.cond[j] == OPPOSE and self.activation_val[j] > self.threshold_sig:
                    best_oppose = j

        var arrestee = best_active
        if arrestee < 0:
            arrestee = best_oppose
        if arrestee >= 0:
            self.rng[sec_idx] = lcg_next(self.rng[sec_idx])
            self.jail_sent[arrestee] = lcg_int(self.rng[sec_idx], self.max_jail)
            self.cond[arrestee] = JAILED


# ============================================================
# Main
# ============================================================

def main():
    print("=== Cascade Batch Simulation (Mojo) ===")
    print("GPU available:", has_accelerator())

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
    var total = len(seeds) * len(epsilons) * len(sec_densities)
    print("Running", total, "simulations x", num_steps, "steps")
    print()

    var t_start = perf_counter_ns()
    var idx = 0
    for si in range(len(seeds)):
        for ei in range(len(epsilons)):
            for di in range(len(sec_densities)):
                var sim = SimState(
                    seed_val=seeds[si],
                    citizen_density=Float32(0.7),
                    sec_density=sec_densities[di],
                    pp_mean=Float32(0.0),
                    model_eps=epsilons[ei],
                    threshold=Float32(2.94444),
                )

                for s in range(num_steps):
                    sim.step()

                print(
                    "Sim", idx,
                    "seed=", seeds[si],
                    "eps=", epsilons[ei],
                    "sd=", sec_densities[di],
                    "active=", sim.active_count,
                    "support=", sim.support_count,
                    "oppose=", sim.oppose_count,
                    "jail=", sim.jail_count,
                    "rev=", sim.revolution,
                )
                idx += 1

    var elapsed_ns = perf_counter_ns() - t_start
    var elapsed_s = Float64(elapsed_ns) / 1_000_000_000.0
    print()
    print("Done:", idx, "simulations in", elapsed_s, "seconds")
    print("Throughput:", Float64(idx) / elapsed_s, "sims/sec")
    print("Per-sim:", elapsed_s / Float64(idx), "s")
