from __future__ import annotations

import io
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python-core-simulation"))

from cascade_core import Citizen, Condition, ResistanceCascade, Security  # noqa: E402


def _run_tests() -> None:
    test_initialization_uses_core_gaussian_mechanics()
    test_visible_counts_self_count_and_exclude_same_cell_occupants()
    test_decision_function_uses_single_draw_active_then_oppose()
    test_float32_numeric_mode_uses_gpu_style_rounding()
    test_step_applies_simultaneous_citizen_updates_before_movement()
    test_security_arrests_active_neighbor_and_jails_with_inclusive_sentence()
    test_full_per_agent_trace_output_is_collected_and_writable()
    test_revolution_metric_counts_active_and_jailed_citizens()



def test_initialization_uses_core_gaussian_mechanics() -> None:
    # Arrange / Act
    sim = ResistanceCascade(
        width=4,
        height=4,
        citizen_density=0.5,
        security_density=0.25,
        private_preference_distribution_mean=-0.5,
        standard_deviation=1.0,
        epsilon=0.5,
        threshold=2.94444,
        seed=123,
    )

    # Assert
    assert len(sim.citizens) == 8
    assert len(sim.security) == 4
    first = sim.citizens[0]
    assert first.condition is Condition.SUPPORT
    assert first.oppose_threshold <= first.active_threshold
    assert math.isclose(first.epsilon_probability, sim.sigmoid(first.epsilon))


def test_visible_counts_self_count_and_exclude_same_cell_occupants() -> None:
    # Arrange
    sim = ResistanceCascade(width=5, height=5, citizen_density=0, security_density=0, seed=1)
    sim.citizens = [
        Citizen(position=(2, 2), private_preference=0, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2),
        Citizen(position=(2, 2), private_preference=0, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2, condition=Condition.OPPOSE),
        Citizen(position=(3, 2), private_preference=0, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2, condition=Condition.ACTIVE),
    ]
    sim.security = [Security(position=(2, 3), private_preference=0)]

    # Act
    counts = sim.visible_counts(0)

    # Assert
    assert counts.active == 2
    assert counts.oppose == 0
    assert counts.support == 1
    assert counts.security == 1


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def test_decision_function_uses_single_draw_active_then_oppose() -> None:
    # Arrange
    sim = ResistanceCascade(width=5, height=5, citizen_density=0, security_density=0, seed=1)
    sim.citizens = [
        Citizen(position=(2, 2), private_preference=-5, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2),
    ]

    # Act
    high_draw = sim.decision_for(0, activation_draw=0.99)
    low_draw = sim.decision_for(0, activation_draw=0.01)

    # Assert
    assert high_draw.condition is Condition.OPPOSE
    assert low_draw.condition is Condition.ACTIVE
    assert low_draw.oppose_level > low_draw.active_level


def test_float32_numeric_mode_uses_gpu_style_rounding() -> None:
    # Arrange
    sim = ResistanceCascade(width=5, height=5, citizen_density=0, security_density=0, seed=1, numeric_mode="float32")
    sim.citizens = [
        Citizen(
            position=(2, 2),
            private_preference=_f32(-0.25),
            epsilon=_f32(0.5),
            epsilon_probability=_f32(1.0 / (1.0 + math.exp(-_f32(0.5)))),
            oppose_threshold=_f32(1.25),
            active_threshold=_f32(2.5),
        ),
        Citizen(
            position=(3, 2),
            private_preference=_f32(0.0),
            epsilon=_f32(0.0),
            epsilon_probability=_f32(0.5),
            oppose_threshold=_f32(1.0),
            active_threshold=_f32(2.0),
            condition=Condition.ACTIVE,
        ),
    ]

    # Act
    decision = sim.decision_for(0, activation_draw=0.99)

    # Assert
    active_ratio = _f32(_f32(2.0) / _f32(1.0))
    exponent = _f32(_f32(1.0) / _f32(_f32(_f32(0.5) * _f32(0.5)) + _f32(1.0)))
    perception = _f32(_f32(2.0) ** exponent)
    expected_opinion = _f32(_f32(0.25) + _f32(perception * active_ratio))
    assert decision.opinion == expected_opinion
    assert decision.activation == _f32(_f32(1.0) / _f32(_f32(1.0) + _f32(math.exp(_f32(-expected_opinion)))))


def test_step_applies_simultaneous_citizen_updates_before_movement() -> None:
    # Arrange
    sim = ResistanceCascade(width=5, height=5, citizen_density=0, security_density=0, seed=7, movement=False)
    sim.citizens = [
        Citizen(position=(2, 2), private_preference=-5, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2),
        Citizen(position=(3, 2), private_preference=5, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2),
    ]

    # Act
    sim.step(activation_draws=[0.01, 0.99])

    # Assert
    assert [citizen.condition for citizen in sim.citizens] == [Condition.ACTIVE, Condition.SUPPORT]
    assert [citizen.position for citizen in sim.citizens] == [(2, 2), (3, 2)]


def test_security_arrests_active_neighbor_and_jails_with_inclusive_sentence() -> None:
    # Arrange
    sim = ResistanceCascade(width=5, height=5, citizen_density=0, security_density=0, seed=3, movement=False, max_jail_term=0)
    sim.citizens = [
        Citizen(position=(2, 2), private_preference=0, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2, condition=Condition.ACTIVE),
    ]
    sim.security = [Security(position=(2, 3), private_preference=0)]

    # Act
    sim.security_arrest_and_move()

    # Assert
    assert sim.citizens[0].condition is Condition.JAILED
    assert sim.citizens[0].jail_sentence == 0
    assert sim.citizens[0].position is None


def test_full_per_agent_trace_output_is_collected_and_writable() -> None:
    # Arrange
    sim = ResistanceCascade(width=4, height=4, citizen_density=0.25, security_density=0.125, seed=11, movement=False)
    initial_rows = len(sim.citizens) + len(sim.security)

    # Act
    sim.step()
    output = io.StringIO()
    sim.write_trace(output)

    # Assert
    assert len(sim.trace) == initial_rows * 2
    assert {row.agent_type for row in sim.trace} == {"Citizen", "Security"}
    citizen_row = next(row for row in sim.trace if row.agent_type == "Citizen")
    assert citizen_row.private_preference is not None
    assert citizen_row.epsilon is not None
    assert citizen_row.active_threshold is not None
    assert output.getvalue().startswith("step,agent_id,agent_type")


def test_revolution_metric_counts_active_and_jailed_citizens() -> None:
    # Arrange
    sim = ResistanceCascade(width=10, height=10, citizen_density=0, security_density=0, seed=1)
    sim.citizens = [
        *[Citizen(position=(0, 0), private_preference=0, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2, condition=Condition.ACTIVE) for _ in range(94)],
        *[Citizen(position=(0, 0), private_preference=0, epsilon=0, epsilon_probability=0.5, oppose_threshold=1, active_threshold=2, condition=Condition.SUPPORT) for _ in range(6)],
    ]

    # Act
    sim.update_revolution_status()

    # Assert
    assert not sim.revolution

    # Act
    sim.citizens[-1].condition = Condition.ACTIVE
    sim.update_revolution_status()

    # Assert
    assert sim.revolution


if __name__ == "__main__":
    _run_tests()
