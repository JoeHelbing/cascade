import math
from .random_walker import RandomWalker


class Citizen(RandomWalker):
    """
    Citizen agent that looks at neighbors and decides whether to activate
    based on number of active neighbors and its own activation level.
    """

    _is_citizen = True
    _is_security = False

    def __init__(
        self,
        unique_id,
        model,
        pos,
        vision,
        private_preference,
        epsilon,
        epsilon_probability,
        oppose_threshold,
        active_threshold,
    ):
        super().__init__(unique_id, model, pos)
        self.vision = vision

        # simultaneous activation attributes
        self._update_condition = None

        # agent personality attributes
        self.private_preference = private_preference
        self.epsilon = epsilon
        self.epsilon_probability = epsilon_probability
        self.oppose_threshold = oppose_threshold
        self.active_threshold = active_threshold

        # agent memory attributes
        self.flip = None
        self.ever_flipped = False
        self.condition = "Support"
        self.perception = None
        self.arrest_prob = None
        self.actives_in_vision = 1
        self.opposed_in_vision = 0
        self.support_in_vision = 0
        self.security_in_vision = 0
        self.opinion = None
        self.activation = None
        self.active_level = None
        self.oppose_level = None

        # agent jail attributes
        self.jail_sentence = 0

    # Alias for typo compatibility with original code
    @property
    def opposes_in_vision(self):
        return self.opposed_in_vision

    def step(self):
        """Decide whether to activate, then move if applicable."""
        self.flip = False

        if self.jail_sentence > 0 or self.condition == "Jailed":
            return

        # Combined neighbor update + count in one grid pass
        grid = self.model.grid
        neighborhood = grid.get_neighborhood(self.pos, moore=True, radius=self.vision)
        grid_data = grid._grid

        actives = 1
        opposed = 0
        support = 1
        security = 0
        neighbors = []
        extend = neighbors.extend

        for x, y in neighborhood:
            cell = grid_data[x][y]
            if cell:
                extend(cell)

        for n in neighbors:
            if n._is_citizen:
                cond = n.condition
                if cond == "Active":
                    actives += 1
                elif cond == "Oppose":
                    opposed += 1
                elif cond == "Support":
                    support += 1
            else:
                security += 1

        self.neighbors = neighbors
        self.actives_in_vision = actives
        self.opposed_in_vision = opposed
        self.support_in_vision = support
        self.security_in_vision = security

        self.determine_condition()

    def advance(self):
        """Advance the citizen to the next step of the model."""
        if self.jail_sentence > 0:
            self.jail_sentence -= 1
            return
        elif self.jail_sentence <= 0 and self.condition == "Jailed":
            self.pos = self.random.choice(list(self.model.grid.empties))
            self.model.grid.place_agent(self, self.pos)
            self.condition = "Support"

        self.condition = self._update_condition
        self.random_move()

    def count_neigbhors(self):
        """Count the number of neighbors of each type."""
        self.actives_in_vision = 1
        self.opposed_in_vision = 0
        self.support_in_vision = 1
        self.security_in_vision = 0

        for neighbor in self.neighbors:
            if neighbor._is_citizen:
                cond = neighbor.condition
                if cond == "Active":
                    self.actives_in_vision += 1
                elif cond == "Oppose":
                    self.opposed_in_vision += 1
                elif cond == "Support":
                    self.support_in_vision += 1
            else:
                self.security_in_vision += 1

    def determine_condition(self):
        """
        Activation function that determines whether citizen will support
        or activate. Neighbor counts are pre-computed in step().
        """
        # ratio of active and oppose to citizens in vision
        self.active_ratio = (
            self.actives_in_vision + self.opposed_in_vision
        ) / self.support_in_vision

        # perceptions of support/oppose/active
        self.perception = (
            self.actives_in_vision + self.opposed_in_vision * self.epsilon_probability
        ) ** ((self.epsilon**2 + 1) ** -1)

        # Probability of arrest P
        self.arrest_prob = 1 - math.exp(
            -2.3
            * (self.security_in_vision / (self.actives_in_vision))
            * (2 * self.epsilon_probability)
        )

        # Calculate opinion and determine condition
        self.opinion = (
            (-1 * self.private_preference)
            + (self.perception * self.active_ratio)
        )

        # uniform random activation 0.0 - 1.0
        random_activation = self.model.random.random()

        # calculate activation levels — inline sigmoid for speed
        opinion = self.opinion
        arrest_prob = self.arrest_prob
        self.activation = 1.0 / (1.0 + math.exp(-opinion))
        self.active_level = (
            1.0 / (1.0 + math.exp(-(opinion - self.active_threshold))) - arrest_prob
        )
        self.oppose_level = (
            1.0 / (1.0 + math.exp(-(opinion - self.oppose_threshold))) - arrest_prob
        )

        # assign condition by activation level
        if self.active_level > random_activation:
            if self._update_condition != "Active":
                self.flip = True
                self.ever_flipped = True
            self._update_condition = "Active"
        elif self.oppose_level > random_activation:
            self._update_condition = "Oppose"
        else:
            self._update_condition = "Support"


class Security(RandomWalker):
    """
    Security agent that arrests active neighbors.
    """

    _is_citizen = False
    _is_security = True

    def __init__(self, unique_id, model, pos, vision, private_preference):
        super().__init__(unique_id, model, pos)
        self.pos = pos
        self.vision = vision
        self.condition = "Security"
        self.memory = None
        self.defected = False
        self._new_identity = None
        self.private_preference = private_preference

        # attributes for data collection compatibility
        self.opinion = None
        self.activation = None
        self.risk_aversion = None
        self.oppose_threshold = None
        self.active_threshold = None
        self.epsilon = None
        self.epsilon_probability = None
        self.jail_sentence = None
        self.flip = None
        self.ever_flipped = None
        self.perception = None
        self.arrest_prob = None
        self.actives_in_vision = None
        self.opposed_in_vision = None
        self.support_in_vision = None
        self.security_in_vision = None
        self.active_level = None
        self.oppose_level = None

    # Alias for typo compatibility
    @property
    def opposes_in_vision(self):
        return self.opposed_in_vision

    def step(self):
        """Steps for security class to determine behavior."""
        pass

    def advance(self):
        """Advance for security class to determine behavior."""
        self.arrest()
        self.random_move()

    def arrest(self):
        """Arrests active neighbor."""
        neighbor_cells = self.model.grid.get_neighborhood(self.pos, moore=True)

        active_neighbors = []
        oppose_neighbors = []
        threshold_sig = self.model.threshold_constant_sigmoid
        for neighbor in self.model.grid.get_cell_list_contents(neighbor_cells):
            if not neighbor._is_citizen:
                continue
            cond = neighbor.condition
            if cond == "Active":
                active_neighbors.append(neighbor)
            elif cond == "Oppose" and neighbor.activation is not None and neighbor.activation > threshold_sig:
                oppose_neighbors.append(neighbor)

        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            sentence = self.random.randint(0, self.model.max_jail_term)
            arrestee.jail_sentence = sentence
            arrestee.condition = "Jailed"
            self.model.grid.remove_agent(arrestee)
        elif oppose_neighbors:
            arrestee = self.random.choice(oppose_neighbors)
            sentence = self.random.randint(0, self.model.max_jail_term)
            arrestee.jail_sentence = sentence
            arrestee.condition = "Jailed"
            self.model.grid.remove_agent(arrestee)
