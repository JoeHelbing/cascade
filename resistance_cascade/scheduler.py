"""Lightweight two-phase simultaneous activation scheduler replacing mesa.time."""

from collections import defaultdict


class SimultaneousActivationByTypeFiltered:
    """
    A scheduler that executes agents in two phases (step + advance)
    and tracks agents by type. Drop-in replacement for the mesa-based version.
    """

    def __init__(self, model):
        self.model = model
        self._agents = {}
        self.agents_by_type = defaultdict(dict)
        self._steps = 0

    @property
    def agents(self):
        return list(self._agents.values())

    def add(self, agent):
        """Add an agent to the schedule."""
        self._agents[agent.unique_id] = agent
        agent_class = type(agent)
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent):
        """Remove an agent from the schedule."""
        del self._agents[agent.unique_id]
        agent_class = type(agent)
        del self.agents_by_type[agent_class][agent.unique_id]

    def step(self):
        """Execute one full step: all agents step(), then all agents advance().

        Matches mesa.time.SimultaneousActivation: iterates in insertion order,
        no shuffling. Keys are re-fetched between phases to handle agents
        added/removed during stepping.
        """
        # Phase 1: Decision
        agent_keys = list(self._agents.keys())
        for key in agent_keys:
            self._agents[key].step()

        # Phase 2: Action (recompute keys in case agents were removed)
        agent_keys = list(self._agents.keys())
        for key in agent_keys:
            self._agents[key].advance()

        self._steps += 1

    def get_type_count(self, type_class, filter_func=None):
        """Count agents of a given type, optionally filtered."""
        count = 0
        for agent in self.agents_by_type[type_class].values():
            if filter_func is None or filter_func(agent):
                count += 1
        return count
