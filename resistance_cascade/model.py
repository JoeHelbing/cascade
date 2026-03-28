import math
import logging as log
import random
from .scheduler import SimultaneousActivationByTypeFiltered
from .grid import MultiGrid
from .agent import Citizen, Security


class DataCollector:
    """Lightweight data collector replacing mesa.DataCollector."""

    def __init__(self, model_reporters=None, agent_reporters=None):
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self.model_data = {name: [] for name in self.model_reporters}
        self.agent_data = {name: [] for name in self.agent_reporters}

        # Pre-classify reporters for fast dispatch
        self._callable_reporters = []
        self._attr_reporters = []
        for name, reporter in self.agent_reporters.items():
            if callable(reporter):
                self._callable_reporters.append((name, reporter))
            else:
                self._attr_reporters.append((name, reporter))

    def collect(self, model):
        for name, reporter in self.model_reporters.items():
            self.model_data[name].append(reporter(model))

        agents = model.schedule.agents
        for name, reporter in self._callable_reporters:
            self.agent_data[name].append(
                {a.unique_id: reporter(a) for a in agents}
            )
        for name, attr in self._attr_reporters:
            self.agent_data[name].append(
                {a.unique_id: getattr(a, attr, None) for a in agents}
            )

    def get_model_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.model_data)

    def get_agent_dataframe(self):
        import pandas as pd
        frames = []
        for step_idx in range(len(next(iter(self.agent_data.values())))):
            step_data = {}
            for name in self.agent_reporters:
                step_data[name] = self.agent_data[name][step_idx]
            df = pd.DataFrame(step_data)
            df["Step"] = step_idx
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()


class ResistanceCascade:
    """
    The resistance cascade model. Citizens decide whether to support, oppose,
    or actively resist the regime based on their neighbors' behavior,
    private preferences, and fear of arrest by Security forces.
    """

    def __init__(
        self,
        width=40,
        height=40,
        citizen_vision=7,
        citizen_density=0.7,
        security_density=0.00,
        security_vision=7,
        max_jail_term=100,
        movement=True,
        multiple_agents_per_cell=True,
        private_preference_distribution_mean=0,
        standard_deviation=1,
        epsilon=0.5,
        max_iters=1000,
        threshold=3.66356,
        seed=None,
        random_seed=False,
    ):
        # Set up reproducible RNG matching mesa's behavior
        if random_seed:
            seed = random.randint(0, 1000000)
        self._seed = seed
        self.random = random.Random(seed)

        print(f"Running ResistanceCascade with seed {self._seed}")
        log.info(f"Running ResistanceCascade with seed {self._seed}")

        self.width = width
        self.height = height
        self.movement = movement
        self.multiple_agents_per_cell = multiple_agents_per_cell

        self.citizen_density = citizen_density
        self.citizen_vision = citizen_vision
        self.private_preference_distribution_mean = private_preference_distribution_mean
        self.standard_deviation = standard_deviation
        self.epsilon = epsilon
        self.threshold = threshold
        self.threshold_constant_sigmoid = self.sigmoid(self.threshold)
        self.security_density = security_density
        self.security_vision = security_vision

        self.max_jail_term = max_jail_term
        self.citizen_count = round(self.width * self.height * self.citizen_density)
        self.security_count = round(self.width * self.height * self.security_density)

        self.max_iters = max_iters
        self.iteration = 0
        self.random_seed = random_seed
        self._next_id_counter = 0

        self.schedule = SimultaneousActivationByTypeFiltered(self)
        self.grid = MultiGrid(self.width, self.height, torus=True)

        self.support_count = 0
        self.active_count = 0
        self.oppose_count = 0
        self.revolution = False

        # Create citizens
        for i in range(self.citizen_count):
            pos = None
            if not self.multiple_agents_per_cell and len(self.grid.empties) > 0:
                pos = self.random.choice(list(self.grid.empties))
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                pos = (x, y)

            private_preference = self.random.gauss(
                self.private_preference_distribution_mean, self.standard_deviation
            )
            epsilon = self.random.gauss(0, self.epsilon)
            epsilon_probability = self.sigmoid(epsilon)
            thresholds = [self.random.gauss(self.threshold, epsilon) for _ in range(0, 2)]
            oppose_threshold = min(thresholds)
            active_threshold = max(thresholds)

            citizen = Citizen(
                self.next_id(),
                self,
                pos,
                self.citizen_vision,
                private_preference,
                epsilon,
                epsilon_probability,
                oppose_threshold,
                active_threshold,
            )
            self.grid.place_agent(citizen, pos)
            self.schedule.add(citizen)

        # Create security
        for i in range(self.security_count):
            pos = None
            if not self.multiple_agents_per_cell and len(self.grid.empties) > 0:
                pos = self.random.choice(list(self.grid.empties))
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                pos = (x, y)

            private_preference = self.random.gauss(
                self.private_preference_distribution_mean, self.standard_deviation
            )

            security = Security(
                self.next_id(),
                self,
                pos,
                self.security_vision,
                private_preference,
            )
            self.grid.place_agent(security, pos)
            self.schedule.add(security)

        # Data collectors
        model_reporters = {
            "Seed": self.report_seed,
            "Citizen Count": self.count_citizen,
            "Active Count": self.count_active,
            "Support Count": self.count_support,
            "Oppose Count": self.count_oppose,
            "Jail Count": self.count_jail,
            "Speed of Spread": self.speed_of_spread,
            "Security Density": self.report_security_density,
            "Private Preference": self.report_private_preference,
            "Epsilon": self.report_epsilon,
            "Threshold": self.report_threshold,
            "Revolution": self.report_revolution,
        }
        agent_reporters = {
            "pos": lambda a: getattr(a, "pos", None),
            "condition": lambda a: getattr(a, "condition", None),
            "opinion": lambda a: getattr(a, "opinion", None),
            "activation": lambda a: getattr(a, "activation", None),
            "private_preference": lambda a: getattr(a, "private_preference", None),
            "epsilon": lambda a: getattr(a, "epsilon", None),
            "oppose_threshold": lambda a: getattr(a, "oppose_threshold", None),
            "active_threshold": lambda a: getattr(a, "active_threshold", None),
            "jail_sentence": lambda a: getattr(a, "jail_sentence", None),
            "actives_in_vision": lambda a: getattr(a, "actives_in_vision", None),
            "opposed_in_vision": lambda a: getattr(a, "opposes_in_vision", None),
            "support_in_vision": lambda a: getattr(a, "supports_in_vision", None),
            "security_in_vision": lambda a: getattr(a, "security_in_vision", None),
            "perception": lambda a: getattr(a, "perception", None),
            "arrest_prob": lambda a: getattr(a, "arrest_prob", None),
            "active_level": lambda a: getattr(a, "active_level", None),
            "oppose_level": lambda a: getattr(a, "oppose_level", None),
            "flip": lambda a: getattr(a, "flip", None),
            "ever_flipped": lambda a: getattr(a, "ever_flipped", None),
            "model_seed": lambda a: getattr(a, "dc_seed", None),
            "model_security_density": lambda a: getattr(a, "dc_security_density", None),
            "model_private_preference": lambda a: getattr(a, "dc_private_preference", None),
            "model_epsilon": lambda a: getattr(a, "dc_epsilon", None),
            "model_threshold": lambda a: getattr(a, "dc_threshold", None),
        }
        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        # Set citizen states prior to first step
        for agent in self.schedule.agents_by_type[Citizen].values():
            # Run the combined neighbor update + count + decide logic
            agent.update_neighbors()
            agent.count_neigbhors()
            agent.determine_condition()

        self.running = True
        self.datacollector.collect(self)

    def next_id(self):
        """Generate the next unique agent ID."""
        self._next_id_counter += 1
        return self._next_id_counter

    def step(self):
        """Advance the model by one step and collect data."""
        self.schedule.step()

        active_or_jailed_agents = sum(
            1 for agent in self.schedule.agents
            if type(agent) is Citizen and (agent.condition == "Active" or agent.condition == "Jailed")
        )
        proportion_active_or_jailed = active_or_jailed_agents / self.citizen_count

        if proportion_active_or_jailed >= 0.95:
            log.debug(f"Stop conditiom met at iteration {self.iteration}, Viva la Revolucion!")
            print(f"Stop conditiom met at iteration {self.iteration}, Viva la Revolucion!")
            self.revolution = True
            self.running = False

        self.datacollector.collect(self)

        self.active_count = self.count_active(self)
        self.support_count = self.count_support(self)
        self.oppose_count = self.count_oppose(self)

        self.iteration += 1
        if self.iteration > self.max_iters:
            self.running = False

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def report_seed(model):
        return model._seed

    @staticmethod
    def count_citizen(model):
        return model.citizen_count

    @staticmethod
    def speed_of_spread(model):
        return (
            len([
                agent for agent in model.schedule.agents_by_type[Citizen].values()
                if agent.flip is True
            ])
            / model.citizen_count
        )

    @staticmethod
    def count_active(model):
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Active"
        ])

    @staticmethod
    def count_oppose(model):
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Oppose"
        ])

    @staticmethod
    def count_support(model):
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Support"
        ])

    @staticmethod
    def count_jail(model):
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Jailed"
        ])

    @staticmethod
    def report_security_density(model):
        return model.security_density

    @staticmethod
    def report_private_preference(model):
        return model.private_preference_distribution_mean

    @staticmethod
    def report_epsilon(model):
        return model.epsilon

    @staticmethod
    def report_threshold(model):
        return model.threshold

    @staticmethod
    def report_revolution(model):
        return model.revolution
