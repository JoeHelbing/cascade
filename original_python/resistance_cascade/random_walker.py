from mesa import Agent as mesaAgent


class RandomWalker(mesaAgent):
    """
    Class implementing random walker methods in a generalized manner.

    Not intended to be used on its own, but to inherit its methods to multiple
    other agents.
    """

    def __init__(self, unique_id, model, pos, moore=True):
        """
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate`
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore

        # model parameters because datacollector needs agent level access
        self.dc_private_preference = self.model.private_preference_distribution_mean
        self.dc_security_density = self.model.security_density
        self.dc_epsilon = self.model.epsilon
        self.dc_seed = self.model._seed
        self.dc_threshold = self.model.threshold

    def update_neighbors(self):
        """
        Update the list of neighbors.
        """
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)

    def random_move(self):
        """
        Step one cell in any allowable direction.
        """
        # Pick the next cell from the adjacent cells.
        next_moves = self.model.grid.get_neighborhood(self.pos, self.moore, True)

        # reduce to valid next moves if we don't allow multiple agents per cell
        if not self.model.multiple_agents_per_cell:
            next_moves = [
                empty for empty in next_moves if self.model.grid.is_cell_empty(empty)
            ]

        # If there are no valid moves stay put
        if not next_moves:
            return

        # randomly choose valid move
        next_move = self.random.choice(next_moves)

        # Now move:
        self.model.grid.move_agent(self, next_move)
