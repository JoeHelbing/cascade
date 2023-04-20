import mesa
import math
import logging as log
import numpy as np


class RandomWalker(mesa.Agent):
    """
    Class implementing random walker methods in a generalized manner.

    Not intended to be used on its own, but to inherit its methods to multiple
    other agents.
    """

    def __init__(self, unique_id, model, pos, moore=True):
        """
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
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

        # move towards other active Citizens if object type is Citizen and condition
        # is Active or if object type is Security
        if (isinstance(self, Citizen) and self.ever_flipped) or isinstance(
            self, Security
        ):
            next_moves = self.move_towards(next_moves)

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

    def determine_avg_loc(self):
        """
        Looks at surrounding cells and determines the average location of the
        of active agents in vision.
        """
        # if no neighbors, return self.pos
        if not self.neighborhood:
            return None

        # pull out the positions of active agents in vision
        pos_ag_list = [
            agent.pos for agent in self.neighborhood if agent.condition == "Active"
        ]

        # calculate the average location of active agents in vision
        if len(pos_ag_list) > 0:
            avg_pos = (
                round(sum([pos[0] for pos in pos_ag_list]) / len(pos_ag_list)),
                round(sum([pos[1] for pos in pos_ag_list]) / len(pos_ag_list)),
            )
        # if no active agents in vision, stay put
        else:
            avg_pos = None

        # update memory
        self.memory = avg_pos

    def move_towards(self, next_moves):
        """
        Whittles choices of next moves to only those that move the agent closer
        to the average location of active agents in vision.
        """
        if self.memory is None:
            return next_moves

        closer_moves = [
            move
            for move in next_moves
            if self.distance(move, self.memory) < self.distance(self.pos, self.memory)
        ]
        return closer_moves

    def logit(self, x):
        """
        Logit function
        """
        return math.log(x / (1 - x))

    def distance(self, pos1, pos2):
        """
        Calculates the distance between two points
        """
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


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
        self.opinion = None
        self.activation = None

        # agent memory attributes
        self.network = None
        self.flip = None
        self.ever_flipped = False
        self.memory = None
        self.condition = "Support"

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

        # memorize avg location of acitve agents
        self.memory = self.determine_avg_loc()

        # random movement
        self.random_move()

        if all(self.neighbors) and self.condition == "Support":
            log.debug(f"Agent {self.unique_id} is {self.condition}")

    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )

        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)

    def count_neigbhors(self):
        """
        Count the number of neighbors of each type
        """
        # Initialize count variables
        actives_in_vision = 1
        opposed_in_vision = 0
        support_in_vision = 0
        security_in_vision = 0

        # Loop through neighbors and count agent types
        for active in self.neighbors:
            if isinstance(active, Citizen):
                if active.condition == "Active":
                    actives_in_vision += 1
                elif active.condition == "Oppose":
                    opposed_in_vision += 1
                elif active.condition == "Support":
                    support_in_vision += 1
            elif isinstance(active, Security):
                security_in_vision += 1

        return (
            actives_in_vision,
            opposed_in_vision,
            support_in_vision,
            security_in_vision,
        )

    def determine_condition(self):
        """
        activation function that determines whether citizen will support
        or activate.
        """
        # return count of neighbor types
        (
            actives_in_vision,
            opposed_in_vision,
            support_in_vision,
            security_in_vision,
        ) = self.count_neigbhors()

        # total number of neighbors in vision
        total_in_vision = (
            actives_in_vision
            + opposed_in_vision
            + support_in_vision
            + security_in_vision
        )

        # ratio of active and oppose to citizens in vision
        active_ratio = (actives_in_vision + opposed_in_vision) / total_in_vision

        # perceptions of support/oppose/active
        perception = np.log(actives_in_vision + opposed_in_vision ) / (
            self.epsilon**2 + 0.05
        )

        # Probability of arrest P
        arrest_prob = 1 - np.exp(
            # constant that produces 0.9 at 1 active (self) and 1 security
            -2.3
            # ratio of securtiy to actives where self is active always
            * (security_in_vision / (actives_in_vision))
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
            + (perception * active_ratio)
            # agents expectation of arrest probability as a function of epsilon
            # interacted with expected cost of arrest interacted with epsilon
            - arrest_prob * (self.model.max_jail_term * self.epsilon_probability)
        )

        self.activation = self.model.sigmoid(self.opinion)

        # assign condition by activation level
        if self.oppose_threshold < self.activation < self.active_threshold:
            self._update_condition = "Oppose"
        elif self.activation > self.active_threshold:
            if self._update_condition != "Active":
                self.flip = True
                self.ever_flipped = True
            self._update_condition = "Active"
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

    def step(self):
        """
        Steps for security class to determine behavior
        """
        # random movement
        self.update_neighbors()
        self._new_identity = self.defect()

    def advance(self):
        """
        Advance for security class to determine behavior
        """
        if self.defected:
            return

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

    def defect(self):
        """
        Defects from the from security
        """
        if (
            all(
                [
                    agent.condition == "Active"
                    for agent in self.neighbors
                    if isinstance(agent, Citizen)
                ]
            )
            and self.private_preference < 0
        ):
            # Recreate self as citizen
            # normal distribution of private regime preference
            private_preference = self.random.gauss(
                self.model.private_preference_distribution_mean,
                self.model.standard_deviation,
            )
            # error term for information controlled society
            epsilon = self.random.gauss(0, self.model.epsilon)
            # epsilon error term sigmoid value
            epsilon_probability = self.model.sigmoid(epsilon)
            # threshold for opposition
            oppose_threshold = self.model.sigmoid(self.model.threshold - abs(epsilon))
            # threshold for active
            active_threshold = self.model.sigmoid(self.model.threshold + abs(epsilon))

            citizen = Citizen(
                self.unique_id,
                self.model,
                self.pos,
                self.vision,
                private_preference,
                epsilon,
                epsilon_probability,
                oppose_threshold,
                active_threshold,
            )
            citizen.condition = "Active"
            self.defected = True
            return citizen

    def remove_thyself(self):
        """
        Removes agent from the grid
        """
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

        self.model.grid.place_agent(self._new_identity, self._new_identity.pos)
        self.model.schedule.add(self._new_identity)
