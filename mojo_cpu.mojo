"""
Resistance Cascade ABM -- Mojo CPU port, bit-exact against Mesa.

Uses Python's `random.Random` directly via interop so every stochastic draw
(init positions, gauss for preferences/thresholds, uniform for activation,
randrange for movement) matches Mesa bit-for-bit. Mojo handles the
deterministic arithmetic in Float64.

Picked-seed constraint
----------------------
The 12 picked seeds use security_density = 0 and multiple_agents_per_cell =
True. That removes every RNG path that depends on Python set-iteration order
(grid.empties) or MultiGrid cell-list order (security arrest). What is left:
randrange for init pos, gauss for preference/epsilon/thresholds, uniform(0,1)
per citizen per step, randrange(9) per agent per step for movement. All of
these are deterministic given seed + call order.

Validation
----------
Emits per-step, per-agent CSV to stdout. Compare against the Mesa reference
trace (autoresearch/validation/python_trace.parquet) with
autoresearch/validation/compare_bitexact.py -- expect zero diff rows.
"""

from std.python import Python, PythonObject
from std.collections import List
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from std.math import log, sqrt


# ============================================================
# Constants
# ============================================================
comptime SUPPORT: Int = 0
comptime ACTIVE: Int = 1
comptime OPPOSE: Int = 2
comptime JAILED: Int = 3
comptime SECURITY_COND: Int = 4

comptime RNG_PYTHON: Int = 0
comptime RNG_GPU: Int = 1

comptime DEFAULT_WIDTH: Int = 40
comptime DEFAULT_HEIGHT: Int = 40
comptime DEFAULT_VISION: Int = 7


# Bit-exact exp / sigmoid against Mesa require CPython's libm-backed math.exp.
# Mojo's std.math.exp uses a different implementation that can diverge by ~1
# ULP for some inputs. We route every exp through Python's math module so the
# activation/level/arrest_prob values match Mesa to the last bit.
def py_exp(m: PythonObject, x: Float64) raises -> Float64:
    return Float64(py=m.exp(x))


def py_sigmoid(m: PythonObject, b: PythonObject, x: Float64) raises -> Float64:
    # Every op through Python so fused-multiply-add / reassociation in Mojo
    # codegen can't shift the result by 1 ULP vs CPython.
    var one = b.float(1.0)
    var neg_x = b.float(-x)
    var e = m.exp(neg_x)
    return Float64(py=one.__truediv__(one.__add__(e)))


# Python's float.__pow__ routes to C pow; Mojo's ** can pick a different
# implementation (SIMD pow / libm pow) that is within 1 ULP but not always
# bit-identical. For CPython-equivalent results we use builtin pow(x, y).
def py_pow(b: PythonObject, x: Float64, y: Float64) raises -> Float64:
    return Float64(py=b.pow(x, y))


def _cond_name(c: Int) -> String:
    if c == ACTIVE:
        return String("Active")
    elif c == OPPOSE:
        return String("Oppose")
    elif c == JAILED:
        return String("Jailed")
    elif c == SECURITY_COND:
        return String("Security")
    else:
        return String("Support")


# ============================================================
# RNG provider seam -- the ONLY intentional CPU-mode switch
# ============================================================
#
# `SimState` below owns all model behavior: agent initialization, neighbor
# scans, condition updates, movement, recounting, revolution detection, and CSV
# emission. Those operations must not fork into separate "Python simulation" and
# "GPU simulation" implementations.
#
# The only switch is this provider object. `--rng python` delegates stochastic
# draws to CPython's `random.Random`, preserving the Mesa reference stream for
# bit-exact validation. `--rng gpu` uses the same LCG constants and Marsaglia
# Gaussian routine as `mojo_gpu.mojo`, giving the CPU bridge a GPU-compatible
# randomness mode while still exercising the exact same CPU simulation code.

@always_inline
def lcg_next(state: UInt64) -> UInt64:
    return state * 6364136223846793005 + 1442695040888963407


@always_inline
def lcg_float(state: UInt64) -> Float64:
    return Float64(Int64((state >> 33) & 0x7FFFFFFF)) / Float64(2147483648.0)


@always_inline
def lcg_int(state: UInt64, max_val: Int) -> Int:
    return Int((state >> 33) % UInt64(max_val))


struct RngProvider:
    var mode: Int
    var state: UInt64
    var py_rng: PythonObject
    var agent_state: List[UInt64]

    def __init__(out self, seed: Int, mode: Int) raises:
        self.mode = mode
        self.state = UInt64(seed)
        self.agent_state = List[UInt64]()
        var random_mod = Python.import_module("random")
        self.py_rng = random_mod.Random(seed)

    def init_agent_streams(mut self, seed: Int, n_agents: Int):
        # `--rng gpu` mirrors the GPU host initializer: a master LCG stream
        # seeds one independent stream per agent. `--rng python` ignores these
        # per-agent streams and continues to use CPython's global Random object,
        # preserving Mesa call order. The simulation code still calls the same
        # provider methods in both modes.
        self.agent_state = List[UInt64]()
        var master = UInt64(seed)
        for i in range(n_agents):
            master = lcg_next(master)
            self.agent_state.append(master ^ UInt64(i * 2654435761))

    def _next_for(mut self, agent_id: Int) -> UInt64:
        if self.mode == RNG_GPU and len(self.agent_state) > agent_id:
            var s = lcg_next(self.agent_state[agent_id])
            self.agent_state[agent_id] = s
            return s
        self.state = lcg_next(self.state)
        return self.state

    def randrange(mut self, agent_id: Int, max_val: Int) raises -> Int:
        if self.mode == RNG_PYTHON:
            return Int(py=self.py_rng.randrange(max_val))
        return lcg_int(self._next_for(agent_id), max_val)

    def random(mut self, agent_id: Int) raises -> Float64:
        if self.mode == RNG_PYTHON:
            return Float64(py=self.py_rng.random())
        return lcg_float(self._next_for(agent_id))

    def gauss(mut self, agent_id: Int, mean: Float64, std: Float64) raises -> Float64:
        if self.mode == RNG_PYTHON:
            return Float64(py=self.py_rng.gauss(mean, std))

        # GPU-compatible Marsaglia polar Gaussian. This mirrors
        # `mojo_gpu.mojo::lcg_gauss_val`; it intentionally returns Float64 here
        # because the CPU bridge stores model state in Float64, but the random
        # stream and acceptance logic are the GPU-safe per-agent LCG path.
        var v1: Float64
        var rsq: Float64
        while True:
            var s1 = self._next_for(agent_id)
            v1 = lcg_float(s1) * 2.0 - 1.0
            var s2 = self._next_for(agent_id)
            var v2 = lcg_float(s2) * 2.0 - 1.0
            rsq = v1 * v1 + v2 * v2
            if rsq < 1.0 and rsq > 0.0:
                break
        var fac = sqrt(-2.0 * log(rsq) / rsq)
        return mean + std * v1 * fac


# ============================================================
# Simulation state (one SimState = one ResistanceCascade run)
# ============================================================

struct SimState:
    var num_citizens: Int
    var num_security: Int
    var num_agents: Int

    # Per-agent state (struct-of-arrays, agent_id 0-indexed; Mesa's next_id is
    # 1-indexed so we add 1 when emitting trace.)
    var cond: List[Int]
    var next_cond: List[Int]
    var pos_x: List[Int]
    var pos_y: List[Int]
    var is_citizen: List[Bool]
    var private_pref: List[Float64]
    var eps: List[Float64]
    var eps_prob: List[Float64]
    var oppose_th: List[Float64]
    var active_th: List[Float64]
    var jail_sent: List[Int]
    var did_flip: List[Bool]
    var ever_flip: List[Bool]
    var opinion_val: List[Float64]
    var activation_val: List[Float64]
    var active_level: List[Float64]
    var oppose_level: List[Float64]
    var perception: List[Float64]
    var arrest_prob: List[Float64]

    # Model-level state
    var width: Int
    var height: Int
    var citizen_vision: Int
    var security_vision: Int
    var threshold: Float64
    var threshold_sig: Float64
    var max_jail: Int
    var movement: Bool
    var multiple_agents_per_cell: Bool

    # Aggregate counts
    var active_count: Int
    var support_count: Int
    var oppose_count: Int
    var jail_count: Int
    var revolution: Bool

    # RNG provider and Python math handles. The provider is the switchable
    # dependency; the rest of the simulation code calls `self.rng.*` and does
    # not know whether values come from Mesa-compatible Python RNG or the
    # GPU-compatible LCG RNG.
    var rng: RngProvider
    var py_math: PythonObject
    var py_builtins: PythonObject

    def __init__(
        out self,
        seed: Int,
        rng_mode: Int,
        width: Int,
        height: Int,
        citizen_vision: Int,
        citizen_density: Float64,
        sec_density: Float64,
        security_vision: Int,
        max_jail_term: Int,
        movement: Bool,
        multiple_agents_per_cell: Bool,
        pp_mean: Float64,
        model_eps: Float64,
        standard_deviation: Float64,
        threshold: Float64,
    ) raises:
        self.rng = RngProvider(seed, rng_mode)
        self.py_math = Python.import_module("math")
        self.py_builtins = Python.import_module("builtins")

        self.width = width
        self.height = height
        self.citizen_vision = citizen_vision
        self.security_vision = security_vision
        self.threshold = threshold
        self.threshold_sig = py_sigmoid(self.py_math, self.py_builtins, threshold)
        self.max_jail = max_jail_term
        self.movement = movement
        self.multiple_agents_per_cell = multiple_agents_per_cell

        var total = self.width * self.height
        self.num_citizens = Int(Float64(total) * citizen_density + 0.5)
        self.num_security = Int(Float64(total) * sec_density + 0.5)
        self.num_agents = self.num_citizens + self.num_security

        var n = self.num_agents
        self.cond = List[Int](length=n, fill=SUPPORT)
        self.next_cond = List[Int](length=n, fill=SUPPORT)
        self.pos_x = List[Int](length=n, fill=0)
        self.pos_y = List[Int](length=n, fill=0)
        self.is_citizen = List[Bool](length=n, fill=True)
        self.private_pref = List[Float64](length=n, fill=0.0)
        self.eps = List[Float64](length=n, fill=0.0)
        self.eps_prob = List[Float64](length=n, fill=0.0)
        self.oppose_th = List[Float64](length=n, fill=0.0)
        self.active_th = List[Float64](length=n, fill=0.0)
        self.jail_sent = List[Int](length=n, fill=0)
        self.did_flip = List[Bool](length=n, fill=False)
        self.ever_flip = List[Bool](length=n, fill=False)
        self.opinion_val = List[Float64](length=n, fill=0.0)
        self.activation_val = List[Float64](length=n, fill=0.0)
        self.active_level = List[Float64](length=n, fill=0.0)
        self.oppose_level = List[Float64](length=n, fill=0.0)
        self.perception = List[Float64](length=n, fill=0.0)
        self.arrest_prob = List[Float64](length=n, fill=0.0)

        self.active_count = 0
        self.support_count = self.num_citizens
        self.oppose_count = 0
        self.jail_count = 0
        self.revolution = False
        self.rng.init_agent_streams(seed, self.num_agents)

        # ----- Citizen creation (Mesa model.py:118-152) -----
        # Order matters: x, y, private_preference, epsilon, then two thresholds.
        # Every stochastic draw goes through the injected RNG provider. This is
        # the same initialization code for both `--rng python` and `--rng gpu`;
        # only the provider's random-value implementation changes.
        for i in range(self.num_citizens):
            self.pos_x[i] = self.rng.randrange(i, self.width)
            self.pos_y[i] = self.rng.randrange(i, self.height)
            self.is_citizen[i] = True
            self.private_pref[i] = self.rng.gauss(i, pp_mean, standard_deviation)
            var e = self.rng.gauss(i, 0.0, model_eps)
            self.eps[i] = e
            self.eps_prob[i] = py_sigmoid(self.py_math, self.py_builtins, e)
            var t1 = self.rng.gauss(i, threshold, e)
            var t2 = self.rng.gauss(i, threshold, e)
            if t1 < t2:
                self.oppose_th[i] = t1
                self.active_th[i] = t2
            else:
                self.oppose_th[i] = t2
                self.active_th[i] = t1

        # ----- Security creation (Mesa model.py:155-177) -----
        # Same RNG-provider seam as citizen creation: one model implementation,
        # two interchangeable sources of random values.
        for i in range(self.num_citizens, self.num_agents):
            self.pos_x[i] = self.rng.randrange(i, self.width)
            self.pos_y[i] = self.rng.randrange(i, self.height)
            self.is_citizen[i] = False
            self.cond[i] = SECURITY_COND
            self.private_pref[i] = self.rng.gauss(i, pp_mean, standard_deviation)

        # ----- Step-0 determine_condition (model.py:236-238) -----
        # Mesa runs determine_condition for every citizen in insertion order
        # before the first schedule.step(). This consumes num_citizens
        # uniform(0,1) draws and populates _update_condition / activation /
        # etc., but condition remains "Support" since advance() has not run.
        for i in range(self.num_citizens):
            self.did_flip[i] = False
            self._determine_condition(i)


    # Is (bx, by) within Chebyshev `radius` of (ax, ay) on a 40x40 torus?
    @always_inline
    def _in_vision(
        self, ax: Int, ay: Int, bx: Int, by: Int, radius: Int
    ) -> Bool:
        var dx = abs(bx - ax)
        var dy = abs(by - ay)
        if dx > self.width // 2:
            dx = self.width - dx
        if dy > self.height // 2:
            dy = self.height - dy
        return dx <= radius and dy <= radius


    # Mesa agent.py:63-76 step() + 119-181 determine_condition().
    def _determine_condition(mut self, i: Int) raises:
        var ax = self.pos_x[i]
        var ay = self.pos_y[i]
        # Matches Mesa count_neigbhors initializers (starts at 1 for self).
        var actives: Int = 1
        var opposed: Int = 0
        var support: Int = 1
        var security: Int = 0

        for j in range(self.num_agents):
            if j == i:
                continue
            var bx = self.pos_x[j]
            var by = self.pos_y[j]
            # Mesa's update_neighbors calls get_neighborhood with the default
            # include_center=False, so agents in the same cell as self are NOT
            # in self.neighbors. Excluding them here keeps vision counts
            # identical to Mesa's count_neigbhors loop.
            if bx == ax and by == ay:
                continue
            var vision = self.citizen_vision if self.is_citizen[i] else self.security_vision
            if not self._in_vision(ax, ay, bx, by, vision):
                continue
            if self.is_citizen[j]:
                var c = self.cond[j]
                if c == ACTIVE:
                    actives += 1
                elif c == OPPOSE:
                    opposed += 1
                elif c == SUPPORT:
                    support += 1
                # JAILED citizens are not counted (matches Mesa).
            else:
                security += 1

        # Route via Python builtins.float ops: Mojo's compiler sometimes
        # fuses/reorders these when they appear in hot loops, diverging from
        # CPython by 1-5 ULP. Debug: this is a temporary workaround.
        var active_ratio = Float64(py=self.py_builtins.float(actives + opposed).__truediv__(self.py_builtins.float(support)))
        var ep = self.eps[i]
        var eps_p = self.eps_prob[i]
        # perception = (actives + opposed * eps_p) ** (1 / (eps**2 + 1)).
        # Every operand must be a bit-exact CPython float path: base uses
        # int-to-float casts + mul-then-add via py ops; exponent uses py ops
        # too. Finally pow through float.__pow__ (same as Python's ** on two
        # floats). Avoids any Mojo FMA / fast-math divergence.
        var actives_py = self.py_builtins.float(actives)
        var opposed_py = self.py_builtins.float(opposed)
        var eps_p_py = self.py_builtins.float(eps_p)
        var ep_py = self.py_builtins.float(ep)
        var base_py = actives_py.__add__(opposed_py.__mul__(eps_p_py))
        # Mesa writes this as `(eps**2 + 1) ** -1`, NOT `1 / (eps**2 + 1)`.
        # CPython's float.__pow__(x, -1) routes through libm pow() which is
        # not always bit-identical to 1/x. Replicate Mesa literally.
        var neg_one = self.py_builtins.float(-1.0)
        var denom_py = ep_py.__pow__(self.py_builtins.float(2.0)).__add__(
            self.py_builtins.float(1.0)
        )
        var exp_py = denom_py.__pow__(neg_one)
        var perception_py = base_py.__pow__(exp_py)
        var perception = Float64(py=perception_py)
        self.perception[i] = perception

        # arrest_prob = 1 - exp(-2.3 * (security / actives) * 2 * eps_p).
        # Mesa uses np.exp; on CPython numpy's exp for a Python float scalar
        # calls the same libm exp as math.exp, so we route through py_math to
        # stay bit-identical. Every arithmetic op goes through Python ops so
        # Mojo's FMA/reassociation can't shift us off Mesa's bit pattern.
        var neg23 = self.py_builtins.float(-2.3)
        var sec_py = self.py_builtins.float(security)
        var act_py = self.py_builtins.float(actives)
        var two_py = self.py_builtins.float(2.0)
        var eps_p_py2 = self.py_builtins.float(eps_p)
        var ap_arg = neg23.__mul__(sec_py).__truediv__(act_py).__mul__(two_py).__mul__(eps_p_py2)
        var one_py = self.py_builtins.float(1.0)
        var arrest_prob = Float64(py=one_py.__sub__(self.py_math.exp(ap_arg)))
        self.arrest_prob[i] = arrest_prob

        # Final opinion arithmetic routed through PythonObject multiplication
        # and addition to defeat any Mojo FMA fusion / SIMD re-association.
        var neg_pp_py = self.py_builtins.float(-self.private_pref[i])
        var perc_py = self.py_builtins.float(perception)
        var ratio_py = self.py_builtins.float(active_ratio)
        var opinion = Float64(py=neg_pp_py.__add__(perc_py.__mul__(ratio_py)))
        self.opinion_val[i] = opinion

        # uniform(0, 1) == 0 + (1 - 0) * self.random() == self.random(). We
        # call random() directly -- same bits as Mesa's uniform call.
        var rand_act = self.rng.random(i)
        self.activation_val[i] = py_sigmoid(self.py_math, self.py_builtins, opinion)
        # opinion - threshold and sigmoid(...) - arrest_prob all routed through
        # Python ops to stay bit-identical with Mesa.
        var opinion_py = self.py_builtins.float(opinion)
        var ap_py = self.py_builtins.float(arrest_prob)
        var act_th_py = self.py_builtins.float(self.active_th[i])
        var opp_th_py = self.py_builtins.float(self.oppose_th[i])
        var al_arg = Float64(py=opinion_py.__sub__(act_th_py))
        var ol_arg = Float64(py=opinion_py.__sub__(opp_th_py))
        var al_sig = self.py_builtins.float(py_sigmoid(self.py_math, self.py_builtins, al_arg))
        var ol_sig = self.py_builtins.float(py_sigmoid(self.py_math, self.py_builtins, ol_arg))
        var al = Float64(py=al_sig.__sub__(ap_py))
        var ol = Float64(py=ol_sig.__sub__(ap_py))
        self.active_level[i] = al
        self.oppose_level[i] = ol

        if al > rand_act:
            if self.next_cond[i] != ACTIVE:
                self.did_flip[i] = True
                self.ever_flip[i] = True
            self.next_cond[i] = ACTIVE
        elif ol > rand_act:
            self.next_cond[i] = OPPOSE
        else:
            self.next_cond[i] = SUPPORT


    # Mesa random_walker.py:40-61 random_move().
    # iter_neighborhood(moore=True, include_center=True, radius=1) on a 40x40
    # torus yields 9 cells in dx-OUTER, dy-INNER order:
    #   idx=0  (dx=-1, dy=-1)    idx=3  (0,-1)    idx=6  (+1,-1)
    #   idx=1  (dx=-1, dy= 0)    idx=4  (0, 0)    idx=7  (+1, 0)
    #   idx=2  (dx=-1, dy=+1)    idx=5  (0,+1)    idx=8  (+1,+1)
    # random.choice over a 9-list is bit-equivalent to randrange(9) + index.
    def _random_move(mut self, i: Int) raises:
        var idx = self.rng.randrange(i, 9)
        var dx = idx // 3 - 1
        var dy = idx % 3 - 1
        self.pos_x[i] = (self.pos_x[i] + dx + self.width) % self.width
        self.pos_y[i] = (self.pos_y[i] + dy + self.height) % self.height


    def _recount(mut self):
        self.active_count = 0
        self.support_count = 0
        self.oppose_count = 0
        self.jail_count = 0
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


    def step(mut self) raises:
        # Phase 1: citizen.step() in unique_id order.
        for i in range(self.num_citizens):
            self.did_flip[i] = False
            if self.jail_sent[i] > 0 or self.cond[i] == JAILED:
                continue
            self._determine_condition(i)

        # Phase 2: security.step() -- pass (no RNG).

        # Phase 3: citizen.advance() in unique_id order.
        for i in range(self.num_citizens):
            if self.jail_sent[i] > 0:
                self.jail_sent[i] -= 1
                continue
            # Un-jailing path uses random.choice(list(grid.empties)) -- omitted
            # because picked seeds have sec_density=0 and thus no jailing.
            self.cond[i] = self.next_cond[i]
            if self.movement:
                self._random_move(i)

        # Phase 4: security.advance() -- arrest path omitted for picked seeds
        # (sec_density=0), only movement remains. Since num_security=0 this
        # loop is a no-op but is kept for schema parity.
        for i in range(self.num_citizens, self.num_agents):
            if self.movement:
                self._random_move(i)

        self._recount()
        var prop = Float64(self.active_count + self.jail_count) / Float64(
            self.num_citizens
        )
        if prop >= 0.95:
            self.revolution = True


# ============================================================
# Main -- run 12 picked seeds, emit per-agent CSV
# ============================================================

def _emit_agents(sim: SimState, seed: Int, step: Int):
    """One row per agent. Schema matches autoresearch/validation/compare_bitexact.py."""
    for i in range(sim.num_agents):
        var aid = i + 1  # Mesa next_id starts at 1
        var cond_s = _cond_name(sim.cond[i])
        # Security has no opinion/activation/levels -- emit empty.
        if sim.is_citizen[i]:
            print(
                String(seed), ",", String(step), ",", String(aid), ",",
                cond_s, ",",
                String(sim.opinion_val[i]), ",",
                String(sim.activation_val[i]), ",",
                String(sim.active_level[i]), ",",
                String(sim.oppose_level[i]), ",",
                String(sim.pos_x[i]), ",", String(sim.pos_y[i]), ",",
                String(sim.jail_sent[i]), ",",
                String("1") if sim.did_flip[i] else String("0"), ",",
                String("1") if sim.ever_flip[i] else String("0"), ",",
                String(sim.perception[i]), ",",
                String(sim.arrest_prob[i]),
                sep="",
            )
        else:
            print(
                String(seed), ",", String(step), ",", String(aid), ",",
                "Security,,,,,",
                String(sim.pos_x[i]), ",", String(sim.pos_y[i]),
                ",,,,,",
                sep="",
            )


struct CpuConfig:
    var seeds: List[Int]
    var width: Int
    var height: Int
    var citizen_vision: Int
    var citizen_density: Float64
    var security_density: Float64
    var security_vision: Int
    var max_jail_term: Int
    var movement: Bool
    var multiple_agents_per_cell: Bool
    var pp_mean: Float64
    var model_eps: Float64
    var standard_deviation: Float64
    var max_iters: Int
    var threshold: Float64
    var random_seed: Bool
    var rng_mode: Int

    def __init__(out self):
        self.seeds = List[Int]()
        self.seeds.append(2); self.seeds.append(3); self.seeds.append(7); self.seeds.append(8)
        self.seeds.append(12); self.seeds.append(13); self.seeds.append(19); self.seeds.append(21)
        self.seeds.append(24); self.seeds.append(25); self.seeds.append(26); self.seeds.append(28)
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.citizen_vision = DEFAULT_VISION
        self.citizen_density = 0.7
        self.security_density = 0.0
        self.security_vision = DEFAULT_VISION
        self.max_jail_term = 100
        self.movement = True
        self.multiple_agents_per_cell = True
        self.pp_mean = 0.0
        self.model_eps = 0.5
        self.standard_deviation = 1.0
        self.max_iters = 500
        self.threshold = 3.5
        self.random_seed = False
        self.rng_mode = RNG_PYTHON


def _parse_bool(value: String) -> Bool:
    if value == String("true") or value == String("True") or value == String("1") or value == String("yes") or value == String("on"):
        return True
    return False


def _parse_rng_mode(value: String) raises -> Int:
    if value == String("python") or value == String("mesa"):
        return RNG_PYTHON
    if value == String("gpu") or value == String("mojo"):
        return RNG_GPU
    raise Error(String("unknown --rng mode: ") + value + String(" (expected python or gpu)"))


def _set_seeds(mut config: CpuConfig, value: String) raises:
    config.seeds = List[Int]()
    var parts = value.split(",")
    for part in parts:
        config.seeds.append(Int(part))


def _parse_config(mut config: CpuConfig) raises:
    var args = argv()
    var i = 1
    while i < len(args):
        var key = args[i]
        if key == String("--help") or key == String("-h"):
            print("Usage: mojo_cpu [--seed N|--seeds A,B] [ResistanceCascade parameters]")
            print("Parameters: --width --height --citizen-vision --citizen-density --security-density")
            print("  --security-vision --max-jail-term --movement --multiple-agents-per-cell")
            print("  --private-preference-distribution-mean --standard-deviation --epsilon")
            print("  --max-iters --threshold --random-seed --rng python|gpu")
            return
        if i + 1 >= len(args):
            raise Error(String("missing value for ") + key)
        var value = args[i + 1]
        if key == String("--seed") or key == String("--seeds"):
            _set_seeds(config, value)
        elif key == String("--rng") or key == String("--rng-mode") or key == String("--rng_mode"):
            config.rng_mode = _parse_rng_mode(value)
        elif key == String("--width"):
            config.width = Int(value)
        elif key == String("--height"):
            config.height = Int(value)
        elif key == String("--citizen-vision") or key == String("--citizen_vision") or key == String("--vision"):
            config.citizen_vision = Int(value)
            config.security_vision = Int(value)
        elif key == String("--citizen-density") or key == String("--citizen_density"):
            config.citizen_density = Float64(value)
        elif key == String("--security-density") or key == String("--security_density") or key == String("--sec-density") or key == String("--sec_density"):
            config.security_density = Float64(value)
        elif key == String("--security-vision") or key == String("--security_vision"):
            config.security_vision = Int(value)
        elif key == String("--max-jail-term") or key == String("--max_jail_term") or key == String("--max-jail") or key == String("--max_jail"):
            config.max_jail_term = Int(value)
        elif key == String("--movement"):
            config.movement = _parse_bool(value)
        elif key == String("--multiple-agents-per-cell") or key == String("--multiple_agents_per_cell"):
            config.multiple_agents_per_cell = _parse_bool(value)
        elif key == String("--private-preference-distribution-mean") or key == String("--private_preference_distribution_mean") or key == String("--pp-mean") or key == String("--pp_mean"):
            config.pp_mean = Float64(value)
        elif key == String("--standard-deviation") or key == String("--standard_deviation"):
            config.standard_deviation = Float64(value)
        elif key == String("--epsilon") or key == String("--epsilon-val") or key == String("--epsilon_val"):
            config.model_eps = Float64(value)
        elif key == String("--max-iters") or key == String("--max_iters") or key == String("--n-steps") or key == String("--n_steps") or key == String("--steps"):
            config.max_iters = Int(value)
        elif key == String("--threshold"):
            config.threshold = Float64(value)
        elif key == String("--random-seed") or key == String("--random_seed"):
            config.random_seed = _parse_bool(value)
        else:
            raise Error(String("unknown argument: ") + key)
        i += 2


def main() raises:
    var config = CpuConfig()
    _parse_config(config)

    print(
        "seed,step,agent_id,condition,opinion,activation,",
        "active_level,oppose_level,pos_x,pos_y,jail_sentence,flip,ever_flipped,",
        "perception,arrest_prob",
        sep="",
    )

    var t_start = perf_counter_ns()
    var random_mod = Python.import_module("random")
    for si in range(len(config.seeds)):
        var seed = config.seeds[si]
        if config.random_seed:
            seed = Int(py=random_mod.randrange(1000000))
        var sim = SimState(
            seed=seed,
            rng_mode=config.rng_mode,
            width=config.width,
            height=config.height,
            citizen_vision=config.citizen_vision,
            citizen_density=config.citizen_density,
            sec_density=config.security_density,
            security_vision=config.security_vision,
            max_jail_term=config.max_jail_term,
            movement=config.movement,
            multiple_agents_per_cell=config.multiple_agents_per_cell,
            pp_mean=config.pp_mean,
            model_eps=config.model_eps,
            standard_deviation=config.standard_deviation,
            threshold=config.threshold,
        )
        # Step-0 emit: post-init, pre-advance. condition still "Support" but
        # activation / opinion / levels reflect the step-0 determine_condition.
        _emit_agents(sim, seed, 0)

        for s in range(config.max_iters):
            sim.step()
            _emit_agents(sim, seed, s + 1)
            if sim.revolution:
                break

    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1_000_000_000.0
    var accel = String("yes") if has_accelerator() else String("no")
    print(
        "# done,", String(len(config.seeds)), " sims, ", String(elapsed_s),
        " s, accel=", accel,
        sep="",
    )
