import mesa

from .model import ResistanceCascade
from .agent import Citizen, Security
from mesa.visualization.UserParam import Slider, NumberInput, Checkbox
from mesa.visualization.modules import ChartModule, TextElement


AGENT_SUPPORT_COLOR = "#648FFF"
AGENT_ACTIVE_COLOR = "#FE6100"
AGENT_OPPOSE_COLOR = "#A020F0"


class ActiveChart(TextElement):
    """Display the current active population."""

    def render(self, model):
        return f"Active Population: {model.active_count}"


class OpposeChart(TextElement):
    """Display the current publicly oppose population."""

    def render(self, model):
        return f"Oppose Population: {model.oppose_count}"


class SupportChart(TextElement):
    """Display the current publicly (regime) supporting population."""

    def render(self, model):
        return f"Supporting Population: {model.support_count}"


active_chart = ActiveChart()
support_chart = SupportChart()
oppose_chart = OpposeChart()

count_chart = ChartModule(
    [
        {"Label": "Support Count", "Color": "#648FFF"},
        {"Label": "Active Count", "Color": "#FE6100"},
        {"Label": "Jail Count", "Color": "#000000"},
        {"Label": "Oppose Count", "Color": "#A020F0"},
    ],
    data_collector_name="datacollector",
)

chart_spread_speed = ChartModule(
    [
        {"Label": "Speed of Spread", "Color": "#000000"},
    ],
    data_collector_name="datacollector",
)


def portrayal(agent):
    if agent is None:
        return

    if type(agent) is Security and agent.defected:
        return

    if type(agent) is Citizen and agent.condition == "Jail":
        return

    portrayal = {
        "Shape": "circle",
        "x": agent.pos[0],
        "y": agent.pos[1],
        "Filled": "true",
    }

    if type(agent) is Citizen:
        if agent.condition == "Active":
            color = AGENT_ACTIVE_COLOR
            layer = 2
        elif agent.condition == "Oppose":
            color = AGENT_OPPOSE_COLOR
            layer = 2
        else:
            color = AGENT_SUPPORT_COLOR
            layer = 0
        radius = 0.5 if agent.condition == "Support" else 0.7
        portrayal["Color"] = color
        portrayal["r"] = radius
        portrayal["Filled"] = "false"
        portrayal["Layer"] = layer

    if type(agent) is Security:
        portrayal["Shape"] = "rect"
        portrayal["w"] = 0.8
        portrayal["h"] = 0.8
        portrayal["Color"] = "#000000"
        portrayal["Layer"] = 1
    return portrayal


model_params = dict(
    height=40,
    width=40,
    citizen_density=Slider("Initial Agent Density", 0.7, 0.0, 0.9, 0.1),
    # citizen_vision=Slider("Citizen Vision", 7, 1, 10, 1),
    # security_vision=Slider("Security Vision", 7, 1, 10, 1),
    security_density=Slider("Security Density", 0.00, 0.0, 0.09, 0.01),
    # max_jail_term=Slider("Maximum Jail Term", 30, 1, 50, 5),
    threshold=Slider("Threshold", 3.66356, 0, 5, 0.1),
    private_preference_distribution_mean=Slider(
        "Mean of Regime Preference", 0, -1, 1, 0.1
    ),
    epsilon=Slider("Epsilon", 1, 0, 1.5, 0.1),
    random_seed=Checkbox("Flip to Using Random Seeds", value=False),
    multiple_agents_per_cell=Checkbox("Multiple Agents Per Cell", value=True),
    seed=NumberInput("User Chosen Fixed Seed", value=42),
)
canvas_element = mesa.visualization.CanvasGrid(portrayal, 40, 40, 480, 480)
server = mesa.visualization.ModularServer(
    ResistanceCascade,
    [
        canvas_element,
        active_chart,
        support_chart,
        oppose_chart,
        count_chart,
        chart_spread_speed,
    ],
    "Resistance Cascade",
    model_params,
)
