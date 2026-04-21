from typing import Type, Callable

from mesa.agent import Agent
from mesa.model import Model

from collections import defaultdict


import mesa

class SimultaneousActivationByTypeFiltered(mesa.time.SimultaneousActivation):
    """
    A scheduler that overrides the get_type_count method to allow for filtering
    of agents by a function before counting.

    Example:
    >>> scheduler = SimultaneousActivationByTypeFiltered(model)
    >>> scheduler.get_type_count(AgentA, lambda agent: agent.some_attribute > 10)
    """
    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.agents_by_type = defaultdict(dict)

    def add(self, agent: Agent) -> None:
        """
        Add an Agent object to the schedule

        Args:
            agent: An Agent to be added to the schedule.
        """

        super().add(agent)
        agent_class: type[Agent] = type(agent)
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        """
        Remove all instances of a given agent from the schedule.
        """

        del self._agents[agent.unique_id]

        agent_class: type[Agent] = type(agent)
        del self.agents_by_type[agent_class][agent.unique_id]


    def get_type_count(
        self,
        type_class: Type[mesa.Agent],
        filter_func: Callable[[mesa.Agent], bool] = None,
    ) -> int:
        """
        Returns the current number of agents of certain type in the queue that satisfy the filter function.
        """
        count = 0
        for agent in self.agents_by_type[type_class].values():
            if filter_func is None or filter_func(agent):
                count += 1
        return count