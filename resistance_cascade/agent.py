import mesa
import math
import logging as log
import numpy as np
from .random_walker import RandomWalker


class Citizen(RandomWalker):
    """
    Citizen agent class that inherits from RandomWalker class. This class
    looks at it's neighbors and decides whether to activate or not based on
    number of active neighbors and it's own activation level.
    """

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
        """
        Attributes and methods inherited from RandomWalker class:
        grid, x, y, moore, update_neighbors, random_move, determine_avg_loc,
        move_towards, sigmoid, logit, distance
        """
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

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        # Set flip to False
        self.flip = False

        if self.jail_sentence > 0 or self.condition == "Jailed":
            return

        # update neighborhood
        self.neighborhood = self.update_neighbors()
        # based on neighborhood determine if support, oppose, or active
        self.determine_condition()

    def advance(self):
        """
        Advance the citizen to the next step of the model.
        """
        # jail sentence
        if self.jail_sentence > 0:
            self.jail_sentence -= 1
            return
        elif self.jail_sentence <= 0 and self.condition == "Jailed":
            self.pos = self.random.choice(list(self.model.grid.empties))
            self.model.grid.place_agent(self, self.pos)
            self.condition = "Support"

        # update condition
        self.condition = self._update_condition

        # random movement
        self.random_move()

    def count_neigbhors(self):
        """
        Count the number of neighbors of each type
        """
        # Initialize count variables
        self.actives_in_vision = 1
        self.opposed_in_vision = 0
        self.support_in_vision = 1
        self.security_in_vision = 0

        # Loop through neighbors and count agent types
        for active in self.neighbors:
            if isinstance(active, Citizen):
                if active.condition == "Active":
                    self.actives_in_vision += 1
                elif active.condition == "Oppose":
                    self.opposed_in_vision += 1
                elif active.condition == "Support":
                    self.support_in_vision += 1
            elif isinstance(active, Security):
                self.security_in_vision += 1

    def determine_condition(self):
        """
        activation function that determines whether citizen will support
        or activate.
        """
        # return count of neighbor types
        self.count_neigbhors()

        # ratio of active and oppose to citizens in vision
        self.active_ratio = (
            self.actives_in_vision + self.opposed_in_vision
        ) / self.support_in_vision

        # perceptions of support/oppose/active
        self.perception = (
            self.actives_in_vision + self.opposed_in_vision * self.epsilon_probability
        ) ** ((self.epsilon**2 + 1) ** -1)

        # Probability of arrest P
        self.arrest_prob = 1 - np.exp(
            # constant that produces 0.9 at 1 active (self) and 1 security
            -2.3
            # ratio of securtiy to actives where self is active always
            * (self.security_in_vision / (self.actives_in_vision))
            # 0 epsilon, ie no error is 0.5 sigmoid probability output
            # where 2 * epsilon is 1.0, aka 1 * probability, aka perfect estimation
            * (2 * self.epsilon_probability)
        )

        # Calculate opinion and determine condition
        self.opinion = (
            # flip private preference so negative regime opinion makes citizen
            # more likely to activate
            (-1 * self.private_preference)
            # perception as a function of the inverse of epsilon squared interacted
            # with the number of actives and opposed in vision
            + (self.perception * self.active_ratio)
            # agents expectation of arrest probability as a function of epsilon
            # interacted with expected cost of arrest interacted with epsilon
        )

        # uniform random activation 0.0 - 1.0
        random_activation = self.model.random.uniform(0.0, 1.0)

        # calculate activation levels
        self.activation = self.model.sigmoid(self.opinion)
        self.active_level = (
            self.model.sigmoid(self.opinion - self.active_threshold) - self.arrest_prob
        )
        self.oppose_level = (
            self.model.sigmoid(self.opinion - self.oppose_threshold) - self.arrest_prob
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
    Security agent class that inherits from RandomWalker class. This class
    looks at it's neighbors and arrests active neighbor

    Attributes and methods inherited from RandomWalker class:
    grid, x, y, moore, update_neighbors, random_move, determine_avg_loc,
    move_towards, sigmoid, logit, distance
    """

    def __init__(self, unique_id, model, pos, vision, private_preference):
        super().__init__(unique_id, model, pos)
        self.pos = pos
        self.vision = vision
        self.condition = "Security"
        self.memory = None
        self.defected = False
        self._new_identity = None
        self.private_preference = private_preference

        # attributes for batch_run and data collection to avoid errors
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

    def step(self):
        """
        Steps for security class to determine behavior
        """
        pass

    def advance(self):
        """
        Advance for security class to determine behavior
        """
        self.update_neighbors()
        self.arrest()
        self.random_move()

    def arrest(self):
        """
        Arrests active neighbor
        """
        neighbor_cells = self.model.grid.get_neighborhood(self.pos, moore=True)

        # collect arrestable neighbors
        active_neighbors = []
        oppose_neighbors = []
        for neighbor in self.model.grid.get_cell_list_contents(neighbor_cells):
            if isinstance(neighbor, Citizen) and neighbor.condition == "Active":
                active_neighbors.append(neighbor)
            elif (
                isinstance(neighbor, Citizen)
                and neighbor.condition == "Oppose"
                and neighbor.activation > self.model.threshold_constant_sigmoid
            ):
                oppose_neighbors.append(neighbor)

        # first arrest active neighbors, then oppose neighbors if no active
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
