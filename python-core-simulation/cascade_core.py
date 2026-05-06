from __future__ import annotations

import csv
import math
import random
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, TextIO


class Condition(str, Enum):
    SUPPORT = "Support"
    OPPOSE = "Oppose"
    ACTIVE = "Active"
    JAILED = "Jailed"
    SECURITY = "Security"


Position = tuple[int, int] | None


@dataclass(slots=True)
class Citizen:
    position: Position
    private_preference: float
    epsilon: float
    epsilon_probability: float
    oppose_threshold: float
    active_threshold: float
    condition: Condition = Condition.SUPPORT
    next_condition: Condition = Condition.SUPPORT
    jail_sentence: int = 0
    opinion: float = 0.0
    activation: float = 0.0
    active_level: float = 0.0
    oppose_level: float = 0.0
    perception: float = 0.0
    arrest_prob: float = 0.0
    active_in_vision: int = 1
    oppose_in_vision: int = 0
    support_in_vision: int = 1
    security_in_vision: int = 0
    flip: bool = False
    ever_flipped: bool = False


@dataclass(slots=True)
class Security:
    position: tuple[int, int]
    private_preference: float
    vision: int = 7
    condition: Condition = Condition.SECURITY


@dataclass(frozen=True, slots=True)
class NeighborCounts:
    active: int
    oppose: int
    support: int
    security: int


@dataclass(frozen=True, slots=True)
class Decision:
    condition: Condition
    opinion: float
    activation: float
    active_level: float
    oppose_level: float
    perception: float
    arrest_prob: float
    counts: NeighborCounts


@dataclass(frozen=True, slots=True)
class TraceRow:
    step: int
    agent_id: int
    agent_type: str
    x: int | None
    y: int | None
    condition: str
    opinion: float | None
    activation: float | None
    private_preference: float
    epsilon: float | None
    oppose_threshold: float | None
    active_threshold: float | None
    jail_sentence: int | None
    active_in_vision: int | None
    oppose_in_vision: int | None
    support_in_vision: int | None
    security_in_vision: int | None
    perception: float | None
    arrest_prob: float | None
    active_level: float | None
    oppose_level: float | None
    flip: bool | None
    ever_flipped: bool | None


TRACE_FIELDS = list(TraceRow.__dataclass_fields__.keys())


def f32(value: float) -> float:
    """Round a Python float through IEEE-754 Float32, matching GPU scalar storage."""
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


class ResistanceCascade:
    def __init__(
        self,
        *,
        width: int = 40,
        height: int = 40,
        citizen_vision: int = 7,
        citizen_density: float = 0.7,
        security_density: float = 0.0,
        security_vision: int = 7,
        max_jail_term: int = 100,
        movement: bool = True,
        private_preference_distribution_mean: float = 0.0,
        standard_deviation: float = 1.0,
        epsilon: float = 0.5,
        threshold: float = 3.66356,
        max_iters: int = 1000,
        seed: int | None = None,
        collect_trace: bool = True,
        numeric_mode: str = "float64",
    ) -> None:
        if numeric_mode not in {"float64", "float32"}:
            raise ValueError("numeric_mode must be 'float64' or 'float32'")
        self.numeric_mode = numeric_mode
        self.width = width
        self.height = height
        self.citizen_vision = citizen_vision
        self.security_vision = security_vision
        self.max_jail_term = max_jail_term
        self.movement = movement
        self.private_preference_distribution_mean = private_preference_distribution_mean
        self.standard_deviation = standard_deviation
        self.epsilon = epsilon
        self.threshold = threshold
        self.threshold_constant_sigmoid = self._sigmoid(threshold)
        self.max_iters = max_iters
        self.iteration = 0
        self.revolution = False
        self.running = True
        self.collect_trace = collect_trace
        self.trace: list[TraceRow] = []
        self.random = random.Random(seed)
        self.seed = seed

        total_cells = self.width * self.height
        citizen_count = round(total_cells * citizen_density)
        security_count = round(total_cells * security_density)

        self.citizens: list[Citizen] = [self._make_citizen() for _ in range(citizen_count)]
        self.security: list[Security] = [self._make_security() for _ in range(security_count)]

        for citizen_id in range(len(self.citizens)):
            decision = self.decision_for(citizen_id)
            self._store_decision(citizen_id, decision)

        self.record_trace()

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _num(self, value: float) -> float:
        return f32(value) if self.numeric_mode == "float32" else float(value)

    def _sigmoid(self, x: float) -> float:
        if self.numeric_mode == "float32":
            x32 = f32(x)
            return f32(f32(1.0) / f32(f32(1.0) + f32(math.exp(f32(-x32)))))
        return self.sigmoid(x)

    @property
    def citizen_count(self) -> int:
        return len(self.citizens)

    def _random_position(self) -> tuple[int, int]:
        return (self.random.randrange(self.width), self.random.randrange(self.height))

    def _make_citizen(self) -> Citizen:
        private_preference = self._num(
            self.random.gauss(
                self.private_preference_distribution_mean,
                self.standard_deviation,
            )
        )
        epsilon = self._num(self.random.gauss(0.0, self.epsilon))
        thresholds = [self._num(self.random.gauss(self.threshold, epsilon)) for _ in range(2)]
        return Citizen(
            position=self._random_position(),
            private_preference=private_preference,
            epsilon=epsilon,
            epsilon_probability=self._sigmoid(epsilon),
            oppose_threshold=min(thresholds),
            active_threshold=max(thresholds),
        )

    def _make_security(self) -> Security:
        return Security(
            position=self._random_position(),
            private_preference=self._num(
                self.random.gauss(
                    self.private_preference_distribution_mean,
                    self.standard_deviation,
                )
            ),
            vision=self.security_vision,
        )

    def visible_counts(self, citizen_id: int) -> NeighborCounts:
        citizen = self.citizens[citizen_id]
        if citizen.position is None:
            return NeighborCounts(active=1, oppose=0, support=1, security=0)

        active = 1
        oppose = 0
        support = 1
        security = 0
        x, y = citizen.position

        for other_id, other in enumerate(self.citizens):
            if other_id == citizen_id or other.position is None:
                continue
            if other.condition is Condition.JAILED:
                continue
            if other.position == citizen.position:
                continue
            if not self._in_vision(x, y, other.position[0], other.position[1], self.citizen_vision):
                continue
            if other.condition is Condition.ACTIVE:
                active += 1
            elif other.condition is Condition.OPPOSE:
                oppose += 1
            elif other.condition is Condition.SUPPORT:
                support += 1

        for officer in self.security:
            if officer.position == citizen.position:
                continue
            if self._in_vision(x, y, officer.position[0], officer.position[1], self.citizen_vision):
                security += 1

        return NeighborCounts(active=active, oppose=oppose, support=support, security=security)

    def _in_vision(self, ax: int, ay: int, bx: int, by: int, vision: int) -> bool:
        dx = abs(ax - bx)
        dy = abs(ay - by)
        torus_dx = min(dx, self.width - dx)
        torus_dy = min(dy, self.height - dy)
        return torus_dx <= vision and torus_dy <= vision

    def decision_for(self, citizen_id: int, activation_draw: float | None = None) -> Decision:
        citizen = self.citizens[citizen_id]
        counts = self.visible_counts(citizen_id)
        if self.numeric_mode == "float32":
            active_ratio = f32(f32(float(counts.active + counts.oppose)) / f32(float(counts.support)))
            base = f32(f32(float(counts.active)) + f32(f32(float(counts.oppose)) * f32(citizen.epsilon_probability)))
            exponent = f32(f32(1.0) / f32(f32(f32(citizen.epsilon) * f32(citizen.epsilon)) + f32(1.0)))
            perception = f32(base**exponent)
            arrest_arg = f32(
                f32(f32(-2.3) * f32(float(counts.security)))
                / f32(float(counts.active))
            )
            arrest_arg = f32(f32(arrest_arg * f32(2.0)) * f32(citizen.epsilon_probability))
            arrest_prob = f32(f32(1.0) - f32(math.exp(arrest_arg)))
            opinion = f32(f32(-citizen.private_preference) + f32(perception * active_ratio))
            activation = self._sigmoid(opinion)
            active_level = f32(self._sigmoid(f32(opinion - citizen.active_threshold)) - arrest_prob)
            oppose_level = f32(self._sigmoid(f32(opinion - citizen.oppose_threshold)) - arrest_prob)
        else:
            active_ratio = (counts.active + counts.oppose) / counts.support
            perception = (counts.active + counts.oppose * citizen.epsilon_probability) ** (
                (citizen.epsilon**2 + 1.0) ** -1
            )
            arrest_prob = 1.0 - math.exp(
                -2.3 * (counts.security / counts.active) * (2.0 * citizen.epsilon_probability)
            )
            opinion = -citizen.private_preference + perception * active_ratio
            activation = self.sigmoid(opinion)
            active_level = self.sigmoid(opinion - citizen.active_threshold) - arrest_prob
            oppose_level = self.sigmoid(opinion - citizen.oppose_threshold) - arrest_prob
        draw = self.random.uniform(0.0, 1.0) if activation_draw is None else activation_draw

        if active_level > draw:
            condition = Condition.ACTIVE
        elif oppose_level > draw:
            condition = Condition.OPPOSE
        else:
            condition = Condition.SUPPORT

        return Decision(
            condition=condition,
            opinion=opinion,
            activation=activation,
            active_level=active_level,
            oppose_level=oppose_level,
            perception=perception,
            arrest_prob=arrest_prob,
            counts=counts,
        )

    def step(self, activation_draws: Iterable[float] | None = None) -> None:
        if not self.running:
            return

        draw_iter = iter(activation_draws) if activation_draws is not None else None
        for citizen in self.citizens:
            citizen.flip = False

        for citizen_id, citizen in enumerate(self.citizens):
            if citizen.jail_sentence > 0 or citizen.condition is Condition.JAILED:
                continue
            draw = next(draw_iter) if draw_iter is not None else None
            decision = self.decision_for(citizen_id, draw)
            self._store_decision(citizen_id, decision)

        for citizen in self.citizens:
            self._advance_citizen(citizen)

        self.security_arrest_and_move()
        self.update_revolution_status()
        self.iteration += 1
        if self.iteration > self.max_iters:
            self.running = False
        self.record_trace()

    def _store_decision(self, citizen_id: int, decision: Decision) -> None:
        citizen = self.citizens[citizen_id]
        citizen.next_condition = decision.condition
        citizen.opinion = decision.opinion
        citizen.activation = decision.activation
        citizen.active_level = decision.active_level
        citizen.oppose_level = decision.oppose_level
        citizen.perception = decision.perception
        citizen.arrest_prob = decision.arrest_prob
        citizen.active_in_vision = decision.counts.active
        citizen.oppose_in_vision = decision.counts.oppose
        citizen.support_in_vision = decision.counts.support
        citizen.security_in_vision = decision.counts.security
        if decision.condition is Condition.ACTIVE and citizen.condition is not Condition.ACTIVE:
            citizen.flip = True
            citizen.ever_flipped = True

    def _advance_citizen(self, citizen: Citizen) -> None:
        if citizen.jail_sentence > 0:
            citizen.jail_sentence -= 1
            return
        if citizen.condition is Condition.JAILED:
            citizen.position = self._random_position()
            citizen.condition = Condition.SUPPORT

        citizen.condition = citizen.next_condition
        if self.movement and citizen.position is not None:
            citizen.position = self._random_move_from(citizen.position)

    def _random_move_from(self, position: tuple[int, int]) -> tuple[int, int]:
        dx = self.random.choice((-1, 0, 1))
        dy = self.random.choice((-1, 0, 1))
        return ((position[0] + dx) % self.width, (position[1] + dy) % self.height)

    def security_arrest_and_move(self) -> None:
        for officer in self.security:
            active_candidates: list[int] = []
            oppose_candidates: list[int] = []
            sx, sy = officer.position
            for citizen_id, citizen in enumerate(self.citizens):
                if citizen.position is None or citizen.position == officer.position:
                    continue
                if not self._in_vision(sx, sy, citizen.position[0], citizen.position[1], 1):
                    continue
                if citizen.condition is Condition.ACTIVE:
                    active_candidates.append(citizen_id)
                elif (
                    citizen.condition is Condition.OPPOSE
                    and citizen.activation > self.threshold_constant_sigmoid
                ):
                    oppose_candidates.append(citizen_id)

            candidates = active_candidates or oppose_candidates
            if candidates:
                arrestee = self.citizens[self.random.choice(candidates)]
                arrestee.jail_sentence = self.random.randint(0, self.max_jail_term)
                arrestee.condition = Condition.JAILED
                arrestee.position = None

            if self.movement:
                officer.position = self._random_move_from(officer.position)

    def update_revolution_status(self) -> None:
        if not self.citizens:
            self.revolution = False
            return
        active_or_jailed = sum(
            1 for citizen in self.citizens if citizen.condition in {Condition.ACTIVE, Condition.JAILED}
        )
        self.revolution = active_or_jailed / len(self.citizens) >= 0.95
        if self.revolution:
            self.running = False

    def count_conditions(self) -> dict[str, int]:
        counts = {condition.value: 0 for condition in Condition}
        for citizen in self.citizens:
            counts[citizen.condition.value] += 1
        counts[Condition.SECURITY.value] = len(self.security)
        return counts

    def record_trace(self) -> None:
        if not self.collect_trace:
            return
        for agent_id, citizen in enumerate(self.citizens):
            x = None if citizen.position is None else citizen.position[0]
            y = None if citizen.position is None else citizen.position[1]
            self.trace.append(
                TraceRow(
                    step=self.iteration,
                    agent_id=agent_id,
                    agent_type="Citizen",
                    x=x,
                    y=y,
                    condition=citizen.condition.value,
                    opinion=citizen.opinion,
                    activation=citizen.activation,
                    private_preference=citizen.private_preference,
                    epsilon=citizen.epsilon,
                    oppose_threshold=citizen.oppose_threshold,
                    active_threshold=citizen.active_threshold,
                    jail_sentence=citizen.jail_sentence,
                    active_in_vision=citizen.active_in_vision,
                    oppose_in_vision=citizen.oppose_in_vision,
                    support_in_vision=citizen.support_in_vision,
                    security_in_vision=citizen.security_in_vision,
                    perception=citizen.perception,
                    arrest_prob=citizen.arrest_prob,
                    active_level=citizen.active_level,
                    oppose_level=citizen.oppose_level,
                    flip=citizen.flip,
                    ever_flipped=citizen.ever_flipped,
                )
            )

        offset = len(self.citizens)
        for security_id, officer in enumerate(self.security):
            self.trace.append(
                TraceRow(
                    step=self.iteration,
                    agent_id=offset + security_id,
                    agent_type="Security",
                    x=officer.position[0],
                    y=officer.position[1],
                    condition=Condition.SECURITY.value,
                    opinion=None,
                    activation=None,
                    private_preference=officer.private_preference,
                    epsilon=None,
                    oppose_threshold=None,
                    active_threshold=None,
                    jail_sentence=None,
                    active_in_vision=None,
                    oppose_in_vision=None,
                    support_in_vision=None,
                    security_in_vision=None,
                    perception=None,
                    arrest_prob=None,
                    active_level=None,
                    oppose_level=None,
                    flip=None,
                    ever_flipped=None,
                )
            )

    def write_trace_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="") as output:
            self.write_trace(output)

    def write_trace(self, output: TextIO) -> None:
        writer = csv.DictWriter(output, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for row in self.trace:
            writer.writerow({field: getattr(row, field) for field in TRACE_FIELDS})

    def run(self, steps: int | None = None) -> None:
        limit = self.max_iters if steps is None else steps
        while self.running and self.iteration < limit:
            self.step()
