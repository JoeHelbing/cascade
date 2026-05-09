"""
Pure Mojo CPU implementation of the Resistance Cascade core simulation.

This intentionally avoids Python interop. It keeps the same model structure as
python-core-simulation/cascade_core.py: Gaussian agent
initialization, Chebyshev/torus visibility, simultaneous citizen decisions,
active-first decision selection, citizen/security movement, security arrests,
and TraceRow-compatible CSV output.
"""

from std.collections import List
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from std.math import exp, log, sqrt


comptime SUPPORT: Int = 0
comptime ACTIVE: Int = 1
comptime OPPOSE: Int = 2
comptime JAILED: Int = 3
comptime SECURITY_COND: Int = 4

comptime DEFAULT_WIDTH: Int = 40
comptime DEFAULT_HEIGHT: Int = 40
comptime DEFAULT_VISION: Int = 7


@always_inline
def sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


@always_inline
def lcg_next(state: UInt64) -> UInt64:
    return state * 6364136223846793005 + 1442695040888963407


struct NativeRng:
    var state: UInt64

    def __init__(out self, seed: Int):
        # Avoid the all-zero-looking first few low bits and keep seed 0 useful.
        self.state = UInt64(seed) ^ UInt64(0x9E3779B97F4A7C15)

    def next_u64(mut self) -> UInt64:
        self.state = lcg_next(self.state)
        return self.state

    def random(mut self) -> Float64:
        # 53 random mantissa bits in [0, 1).
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def randrange(mut self, max_val: Int) -> Int:
        return Int(self.next_u64() % UInt64(max_val))

    def randint(mut self, min_val: Int, max_val: Int) -> Int:
        return min_val + self.randrange(max_val - min_val + 1)

    def gauss(mut self, mean: Float64, std: Float64) -> Float64:
        # Marsaglia polar transform. This is native Mojo, deterministic, and
        # preserves the Gaussian initialization framework of the Python core.
        var v1: Float64
        var rsq: Float64
        while True:
            v1 = self.random() * 2.0 - 1.0
            var v2 = self.random() * 2.0 - 1.0
            rsq = v1 * v1 + v2 * v2
            if rsq > 0.0 and rsq < 1.0:
                break
        var fac = sqrt(-2.0 * log(rsq) / rsq)
        return mean + std * v1 * fac


@always_inline
def _cond_name(c: Int) -> String:
    if c == ACTIVE:
        return String("Active")
    if c == OPPOSE:
        return String("Oppose")
    if c == JAILED:
        return String("Jailed")
    if c == SECURITY_COND:
        return String("Security")
    return String("Support")


struct CoreSim:
    var num_citizens: Int
    var num_security: Int
    var num_agents: Int

    var cond: List[Int]
    var next_cond: List[Int]
    var pos_x: List[Int]
    var pos_y: List[Int]
    var has_position: List[Bool]
    var is_citizen: List[Bool]

    var private_pref: List[Float64]
    var eps: List[Float64]
    var eps_prob: List[Float64]
    var oppose_th: List[Float64]
    var active_th: List[Float64]
    var jail_sent: List[Int]

    var opinion_val: List[Float64]
    var activation_val: List[Float64]
    var active_level: List[Float64]
    var oppose_level: List[Float64]
    var perception: List[Float64]
    var arrest_prob: List[Float64]
    var active_seen: List[Int]
    var oppose_seen: List[Int]
    var support_seen: List[Int]
    var security_seen: List[Int]
    var did_flip: List[Bool]
    var ever_flip: List[Bool]

    var width: Int
    var height: Int
    var citizen_vision: Int
    var security_vision: Int
    var threshold_sig: Float64
    var max_jail: Int
    var movement: Bool
    var revolution: Bool
    var running: Bool
    var iteration: Int
    var max_iters: Int
    var rng: NativeRng

    def __init__(
        out self,
        seed: Int,
        width: Int,
        height: Int,
        citizen_vision: Int,
        citizen_density: Float64,
        security_density: Float64,
        security_vision: Int,
        max_jail_term: Int,
        movement: Bool,
        pp_mean: Float64,
        standard_deviation: Float64,
        model_eps: Float64,
        threshold: Float64,
        max_iters: Int,
    ):
        self.width = width
        self.height = height
        self.citizen_vision = citizen_vision
        self.security_vision = security_vision
        self.max_jail = max_jail_term
        self.movement = movement
        self.threshold_sig = sigmoid(threshold)
        self.max_iters = max_iters
        self.revolution = False
        self.running = True
        self.iteration = 0
        self.rng = NativeRng(seed)

        var cells = width * height
        self.num_citizens = Int(Float64(cells) * citizen_density + 0.5)
        self.num_security = Int(Float64(cells) * security_density + 0.5)
        self.num_agents = self.num_citizens + self.num_security
        var n = self.num_agents

        self.cond = List[Int](length=n, fill=SUPPORT)
        self.next_cond = List[Int](length=n, fill=SUPPORT)
        self.pos_x = List[Int](length=n, fill=0)
        self.pos_y = List[Int](length=n, fill=0)
        self.has_position = List[Bool](length=n, fill=True)
        self.is_citizen = List[Bool](length=n, fill=True)
        self.private_pref = List[Float64](length=n, fill=0.0)
        self.eps = List[Float64](length=n, fill=0.0)
        self.eps_prob = List[Float64](length=n, fill=0.0)
        self.oppose_th = List[Float64](length=n, fill=0.0)
        self.active_th = List[Float64](length=n, fill=0.0)
        self.jail_sent = List[Int](length=n, fill=0)
        self.opinion_val = List[Float64](length=n, fill=0.0)
        self.activation_val = List[Float64](length=n, fill=0.0)
        self.active_level = List[Float64](length=n, fill=0.0)
        self.oppose_level = List[Float64](length=n, fill=0.0)
        self.perception = List[Float64](length=n, fill=0.0)
        self.arrest_prob = List[Float64](length=n, fill=0.0)
        self.active_seen = List[Int](length=n, fill=1)
        self.oppose_seen = List[Int](length=n, fill=0)
        self.support_seen = List[Int](length=n, fill=1)
        self.security_seen = List[Int](length=n, fill=0)
        self.did_flip = List[Bool](length=n, fill=False)
        self.ever_flip = List[Bool](length=n, fill=False)

        for i in range(self.num_citizens):
            self.pos_x[i] = self.rng.randrange(self.width)
            self.pos_y[i] = self.rng.randrange(self.height)
            self.private_pref[i] = self.rng.gauss(pp_mean, standard_deviation)
            var e = self.rng.gauss(0.0, model_eps)
            self.eps[i] = e
            self.eps_prob[i] = sigmoid(e)
            var t1 = self.rng.gauss(threshold, e)
            var t2 = self.rng.gauss(threshold, e)
            if t1 < t2:
                self.oppose_th[i] = t1
                self.active_th[i] = t2
            else:
                self.oppose_th[i] = t2
                self.active_th[i] = t1

        for i in range(self.num_citizens, self.num_agents):
            self.pos_x[i] = self.rng.randrange(self.width)
            self.pos_y[i] = self.rng.randrange(self.height)
            self.is_citizen[i] = False
            self.cond[i] = SECURITY_COND
            self.next_cond[i] = SECURITY_COND
            self.private_pref[i] = self.rng.gauss(pp_mean, standard_deviation)

        for i in range(self.num_citizens):
            self.did_flip[i] = False
            self._determine_condition(i)

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

    def _determine_condition(mut self, i: Int):
        if not self.has_position[i]:
            return
        var ax = self.pos_x[i]
        var ay = self.pos_y[i]
        var actives = 1
        var opposed = 0
        var support = 1
        var security = 0

        for j in range(self.num_citizens):
            if j == i or not self.has_position[j] or self.cond[j] == JAILED:
                continue
            if self.pos_x[j] == ax and self.pos_y[j] == ay:
                continue
            if not self._in_vision(
                ax, ay, self.pos_x[j], self.pos_y[j], self.citizen_vision
            ):
                continue
            if self.cond[j] == ACTIVE:
                actives += 1
            elif self.cond[j] == OPPOSE:
                opposed += 1
            elif self.cond[j] == SUPPORT:
                support += 1

        for j in range(self.num_citizens, self.num_agents):
            if self.pos_x[j] == ax and self.pos_y[j] == ay:
                continue
            if self._in_vision(
                ax, ay, self.pos_x[j], self.pos_y[j], self.citizen_vision
            ):
                security += 1

        self.active_seen[i] = actives
        self.oppose_seen[i] = opposed
        self.support_seen[i] = support
        self.security_seen[i] = security

        var active_ratio = Float64(actives + opposed) / Float64(support)
        var perception_val = (
            Float64(actives) + Float64(opposed) * self.eps_prob[i]
        ) ** ((self.eps[i] ** 2.0 + 1.0) ** -1.0)
        var arrest = 1.0 - exp(
            -2.3
            * (Float64(security) / Float64(actives))
            * (2.0 * self.eps_prob[i])
        )
        var opinion = -self.private_pref[i] + perception_val * active_ratio
        var activation = sigmoid(opinion)
        var al = sigmoid(opinion - self.active_th[i]) - arrest
        var ol = sigmoid(opinion - self.oppose_th[i]) - arrest
        var draw = self.rng.random()

        self.perception[i] = perception_val
        self.arrest_prob[i] = arrest
        self.opinion_val[i] = opinion
        self.activation_val[i] = activation
        self.active_level[i] = al
        self.oppose_level[i] = ol

        if al > draw:
            if self.cond[i] != ACTIVE:
                self.did_flip[i] = True
                self.ever_flip[i] = True
            self.next_cond[i] = ACTIVE
        elif ol > draw:
            self.next_cond[i] = OPPOSE
        else:
            self.next_cond[i] = SUPPORT

    def _random_position(mut self, i: Int):
        self.pos_x[i] = self.rng.randrange(self.width)
        self.pos_y[i] = self.rng.randrange(self.height)
        self.has_position[i] = True

    def _random_move(mut self, i: Int):
        var dx = self.rng.randrange(3) - 1
        var dy = self.rng.randrange(3) - 1
        self.pos_x[i] = (self.pos_x[i] + dx + self.width) % self.width
        self.pos_y[i] = (self.pos_y[i] + dy + self.height) % self.height

    def _advance_citizen(mut self, i: Int):
        if self.jail_sent[i] > 0:
            self.jail_sent[i] -= 1
            return
        if self.cond[i] == JAILED:
            self._random_position(i)
            self.cond[i] = SUPPORT
        self.cond[i] = self.next_cond[i]
        if self.movement and self.has_position[i]:
            self._random_move(i)

    def _security_arrest_and_move(mut self):
        var active_candidates = List[Int]()
        var oppose_candidates = List[Int]()
        for officer in range(self.num_citizens, self.num_agents):
            active_candidates.clear()
            oppose_candidates.clear()
            var sx = self.pos_x[officer]
            var sy = self.pos_y[officer]
            for i in range(self.num_citizens):
                if not self.has_position[i]:
                    continue
                if self.pos_x[i] == sx and self.pos_y[i] == sy:
                    continue
                if not self._in_vision(sx, sy, self.pos_x[i], self.pos_y[i], 1):
                    continue
                if self.cond[i] == ACTIVE:
                    active_candidates.append(i)
                elif (
                    self.cond[i] == OPPOSE
                    and self.activation_val[i] > self.threshold_sig
                ):
                    oppose_candidates.append(i)

            var chosen = -1
            if len(active_candidates) > 0:
                chosen = active_candidates[
                    self.rng.randrange(len(active_candidates))
                ]
            elif len(oppose_candidates) > 0:
                chosen = oppose_candidates[
                    self.rng.randrange(len(oppose_candidates))
                ]
            if chosen >= 0:
                self.jail_sent[chosen] = self.rng.randint(0, self.max_jail)
                self.cond[chosen] = JAILED
                self.next_cond[chosen] = SUPPORT
                self.has_position[chosen] = False

            if self.movement:
                self._random_move(officer)

    def _update_revolution(mut self):
        if self.num_citizens == 0:
            self.revolution = False
            return
        var active_or_jailed = 0
        for i in range(self.num_citizens):
            if self.cond[i] == ACTIVE or self.cond[i] == JAILED:
                active_or_jailed += 1
        self.revolution = (
            Float64(active_or_jailed) / Float64(self.num_citizens) >= 0.95
        )
        if self.revolution:
            self.running = False

    def step(mut self):
        if not self.running:
            return
        for i in range(self.num_citizens):
            self.did_flip[i] = False
        for i in range(self.num_citizens):
            if self.jail_sent[i] > 0 or self.cond[i] == JAILED:
                continue
            self._determine_condition(i)
        for i in range(self.num_citizens):
            self._advance_citizen(i)
        self._security_arrest_and_move()
        self._update_revolution()
        self.iteration += 1
        if self.iteration > self.max_iters:
            self.running = False


struct CoreConfig:
    var seeds: List[Int]
    var width: Int
    var height: Int
    var citizen_vision: Int
    var citizen_density: Float64
    var security_density: Float64
    var security_vision: Int
    var max_jail_term: Int
    var movement: Bool
    var pp_mean: Float64
    var standard_deviation: Float64
    var model_eps: Float64
    var max_iters: Int
    var threshold: Float64
    var random_seed: Bool

    def __init__(out self):
        self.seeds = List[Int]()
        self.seeds.append(2)
        self.seeds.append(3)
        self.seeds.append(7)
        self.seeds.append(8)
        self.seeds.append(12)
        self.seeds.append(13)
        self.seeds.append(19)
        self.seeds.append(21)
        self.seeds.append(24)
        self.seeds.append(25)
        self.seeds.append(26)
        self.seeds.append(28)
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.citizen_vision = DEFAULT_VISION
        self.citizen_density = 0.7
        self.security_density = 0.0
        self.security_vision = DEFAULT_VISION
        self.max_jail_term = 100
        self.movement = True
        self.pp_mean = 0.0
        self.standard_deviation = 1.0
        self.model_eps = 0.5
        self.max_iters = 500
        self.threshold = 3.66356
        self.random_seed = False


def _parse_bool(value: String) -> Bool:
    return (
        value == String("true")
        or value == String("True")
        or value == String("1")
        or value == String("yes")
        or value == String("on")
    )


def _set_seeds(mut config: CoreConfig, value: String) raises:
    config.seeds = List[Int]()
    var parts = value.split(",")
    for part in parts:
        config.seeds.append(Int(part))


def _parse_config(mut config: CoreConfig) raises:
    var args = argv()
    var i = 1
    while i < len(args):
        var key = args[i]
        if key == String("--help") or key == String("-h"):
            print(
                "Usage: core_cpu_mojo [--seed N|--seeds A,B] [ResistanceCascade"
                " parameters]"
            )
            return
        if i + 1 >= len(args):
            raise Error(String("missing value for ") + key)
        var value = args[i + 1]
        if key == String("--seed") or key == String("--seeds"):
            _set_seeds(config, value)
        elif key == String("--width"):
            config.width = Int(value)
        elif key == String("--height"):
            config.height = Int(value)
        elif (
            key == String("--citizen-vision")
            or key == String("--citizen_vision")
            or key == String("--vision")
        ):
            config.citizen_vision = Int(value)
            config.security_vision = Int(value)
        elif key == String("--citizen-density") or key == String(
            "--citizen_density"
        ):
            config.citizen_density = Float64(value)
        elif (
            key == String("--security-density")
            or key == String("--security_density")
            or key == String("--sec-density")
            or key == String("--sec_density")
        ):
            config.security_density = Float64(value)
        elif key == String("--security-vision") or key == String(
            "--security_vision"
        ):
            config.security_vision = Int(value)
        elif (
            key == String("--max-jail-term")
            or key == String("--max_jail_term")
            or key == String("--max-jail")
            or key == String("--max_jail")
        ):
            config.max_jail_term = Int(value)
        elif key == String("--movement"):
            config.movement = _parse_bool(value)
        elif key == String("--multiple-agents-per-cell") or key == String(
            "--multiple_agents_per_cell"
        ):
            # Kept for CLI compatibility. The pure core allows multiple agents per cell.
            pass
        elif (
            key == String("--private-preference-distribution-mean")
            or key == String("--private_preference_distribution_mean")
            or key == String("--pp-mean")
            or key == String("--pp_mean")
        ):
            config.pp_mean = Float64(value)
        elif key == String("--standard-deviation") or key == String(
            "--standard_deviation"
        ):
            config.standard_deviation = Float64(value)
        elif (
            key == String("--epsilon")
            or key == String("--epsilon-val")
            or key == String("--epsilon_val")
        ):
            config.model_eps = Float64(value)
        elif (
            key == String("--max-iters")
            or key == String("--max_iters")
            or key == String("--n-steps")
            or key == String("--n_steps")
            or key == String("--steps")
        ):
            config.max_iters = Int(value)
        elif key == String("--threshold"):
            config.threshold = Float64(value)
        elif key == String("--random-seed") or key == String("--random_seed"):
            config.random_seed = _parse_bool(value)
        elif (
            key == String("--rng")
            or key == String("--rng-mode")
            or key == String("--rng_mode")
        ):
            # Accepted for drop-in comparison scripts; pure Mojo has one native RNG.
            pass
        else:
            raise Error(String("unknown argument: ") + key)
        i += 2


def _emit_agents(sim: CoreSim):
    for i in range(sim.num_agents):
        var cond_s = _cond_name(sim.cond[i])
        var x_s = String(sim.pos_x[i]) if sim.has_position[i] else String("")
        var y_s = String(sim.pos_y[i]) if sim.has_position[i] else String("")
        if sim.is_citizen[i]:
            print(
                String(sim.iteration),
                ",",
                String(i),
                ",Citizen,",
                x_s,
                ",",
                y_s,
                ",",
                cond_s,
                ",",
                String(sim.opinion_val[i]),
                ",",
                String(sim.activation_val[i]),
                ",",
                String(sim.private_pref[i]),
                ",",
                String(sim.eps[i]),
                ",",
                String(sim.oppose_th[i]),
                ",",
                String(sim.active_th[i]),
                ",",
                String(sim.jail_sent[i]),
                ",",
                String(sim.active_seen[i]),
                ",",
                String(sim.oppose_seen[i]),
                ",",
                String(sim.support_seen[i]),
                ",",
                String(sim.security_seen[i]),
                ",",
                String(sim.perception[i]),
                ",",
                String(sim.arrest_prob[i]),
                ",",
                String(sim.active_level[i]),
                ",",
                String(sim.oppose_level[i]),
                ",",
                String("True") if sim.did_flip[i] else String("False"),
                ",",
                String("True") if sim.ever_flip[i] else String("False"),
                sep="",
            )
        else:
            print(
                String(sim.iteration),
                ",",
                String(i),
                ",Security,",
                x_s,
                ",",
                y_s,
                ",Security,,,",
                String(sim.private_pref[i]),
                ",,,,,,,,,,,,,",
                sep="",
            )


def main() raises:
    var config = CoreConfig()
    _parse_config(config)
    print(
        "step,agent_id,agent_type,x,y,condition,opinion,activation,private_preference,",
        "epsilon,oppose_threshold,active_threshold,jail_sentence,active_in_vision,",
        "oppose_in_vision,support_in_vision,security_in_vision,perception,",
        "arrest_prob,active_level,oppose_level,flip,ever_flipped",
        sep="",
    )
    var t_start = perf_counter_ns()
    var seed_rng = NativeRng(Int(t_start & 0x7FFFFFFF))
    for si in range(len(config.seeds)):
        var seed = config.seeds[si]
        if config.random_seed:
            seed = seed_rng.randrange(1000000)
        var sim = CoreSim(
            seed=seed,
            width=config.width,
            height=config.height,
            citizen_vision=config.citizen_vision,
            citizen_density=config.citizen_density,
            security_density=config.security_density,
            security_vision=config.security_vision,
            max_jail_term=config.max_jail_term,
            movement=config.movement,
            pp_mean=config.pp_mean,
            standard_deviation=config.standard_deviation,
            model_eps=config.model_eps,
            threshold=config.threshold,
            max_iters=config.max_iters,
        )
        _emit_agents(sim)
        while sim.running and sim.iteration < config.max_iters:
            sim.step()
            _emit_agents(sim)

    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1_000_000_000.0
    var accel = String("yes") if has_accelerator() else String("no")
    print(
        "# done,",
        String(len(config.seeds)),
        " sims, ",
        String(elapsed_s),
        " s, accel=",
        accel,
        sep="",
    )
