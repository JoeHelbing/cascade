class Agent:
    """Minimal agent base class replacing mesa.Agent."""

    def __init__(self, unique_id, model, pos):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos
        self.random = model.random

        # Model parameters for data collection (agent-level access)
        self.dc_private_preference = model.private_preference_distribution_mean
        self.dc_security_density = model.security_density
        self.dc_epsilon = model.epsilon
        self.dc_seed = model._seed
        self.dc_threshold = model.threshold


class RandomWalker(Agent):
    """
    Base class for agents that can walk randomly on a grid.
    Provides update_neighbors() and random_move().
    """

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

    def update_neighbors(self):
        """Update the list of neighbors within vision radius."""
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)

    def random_move(self):
        """Step one cell in any allowable direction."""
        next_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)

        if not self.model.multiple_agents_per_cell:
            next_moves = [c for c in next_moves if self.model.grid.is_cell_empty(c)]

        if not next_moves:
            return

        next_move = self.random.choice(next_moves)
        self.model.grid.move_agent(self, next_move)
