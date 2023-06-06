# Resistance Cascade Model

Paper can be found at:
[Mentos Regimes: How Individual Uncertainty Affects the Explosive Strength of Resistance Movements—People are the Real Freshmakers
](./Resistance%20Cascade%20Thesis.pdf)

## Summary
This article looks at the empirical data on protests in authoritarian countries as a function of regime type and information control, then constructs an agent based model to examine how the effect of uncertainty can help explain the differences in protest and resistance movements in these differing regime types. The agent based model instantiates two agent types, Citizen and Security, and shows how an inverse relationship between uncertainty with regards acceptable public opposition, the probability of suffering costs, and the ability to accurately perceive local regime support lead to differences in resistance movements. Analysis focuses on the speed of resistance spread between agents as a function of individual agent level uncertainty, and how this affects total resistance size, either full equilibrium flips, i.e. successful revolutions, or protracted unrest. Investigation of empirical data shows reduced frequency of protests in more authoritarian regimes and regimes with higher levels of information control. Modeling dynamics further confirms this behavior and shows a potential connection between lower information control and more frequent but slower spreading, smaller scale resistance events while higher information control is connected with faster inter-agent resistance spread, and larger resistance levels at a reduced frequency.

The data cleaning and analysis R scripts for the section Empirical Observations of Protest Events in Authoritarian Polities are in the 'R_regressions' folder.

The Python code for the agent based model is in the 'resistance_cascade' folder. The model is built using the Mesa framework for agent-based modeling in Python. 

## Files in resistance_cascade/

* ``model.py``: Core model.
* ``server.py``: Sets up the interactive visualization.
* ``agent.py``: Defines the agents Citizen and Security.
* ``schedule.py``: Defines the base schedule SimultaneousActivationByType and the inheriting schedule with added functions.
* ``batch_run_params.json``: Parameters for batch runs, edit this to change parameters for batch runs. The json is loaded in directly as a dictionary to create the grid search matrix.
* ``random_walker.py`` : Defines the base RandomWalker class used by both Citizen and Security which contains the shared functions for movement.


## If Mamba or Conda Not Installed
Install mamba via
```
    $ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    $ bash Mambaforge-$(uname)-$(uname -m).sh

```

## Create Environment
The model can be run in a conda environment. To create the environment, from the repository root directory run:

```
    $ bash create_environment.sh
    $ mamba activate abm
```

## How to Run

To use the model interactively, from the repository root directory run:

```
    $ python run.py
```

The console will print the URL, but you can also open your browser directly to [http://127.0.0.1:8521/](http://127.0.0.1:8521/), choose your parameters, press Reset, then Run.

To use the model in a batch format, edit the parameters you want to run within the json file in ./resistance_cascade/batch_run_parameters.json, then from the repository root directory run:

```
    $ mpirun -np 4 python batch_run.py
```
where 4 can be replaced with the number of cores you want to use for the batch run. The batch run will output a set of model level and individual agent level parquet files for
each individual model run. 

## Model Visualization
<div style="text-align:center">
    <img src="model_analysis/images/step_19_cascade.png"  width="40%">
    <p>Figure: Resistance Cascade Agent Based Model Visualization with Advancing Cascade</p>
</div>

In the model visualization, circles represent Citizen agents in their three *public preference* states: blue being **Support**, purple being **Oppose**, and red being **Active**. Security are represented by the black squares. 

## Model Behavior
The full breakdown of how Citizen activation works can be found in the paper, but a simplified explanation of model behavior is as follows:

The model consists of two types of agent—Citizen agents which are the
primary agents of question and are the subject of most macro-scale model
measures, and Security agents. While this paper is mainly focused on the
complex emergent cascade behaviors of Citizens, the interactions of
Citizens and the state is core to understanding the processes. In
multiple case studies of resistance movements, Security forces act as
the primary foil of Citizens heavily influencing their choices of if,
how, when, and where to activate publicly. Even in scenarios where a
cascade has obviously begun, Security forces have significant ability to
shape events .

The primary agent level attribute of Citizen agents is *private
preference* which derives its theoretical basis from "Now out of Never"
. In the paper, Kuran defines private preference as some internally held
opinion on a regime, either for or against the status quo, which at any
point in time is essentially fixed. Citizens are also defined by a value
*epsilon* which is the primary point of research for this paper.
*Epsilon* is the operationalization of uncertainty across various regime
types and levels of information control. In different regime types,
those of higher or lower information control, how each individual in
that society interacts with regime expectations of what is and is not an
acceptable public opinion, methods of display of those opinions, and the
internal private preferences of one’s neighbors, family, and friends
carries with it a level of uncertainty. How that individual uncertainty
interacts with the environment and the state can describe in part the
differences we see in resistance cascades across these varying regime
types.

The Citizen agents *public preference* or visible state is a function of
the above two exogenous variables. The Citizen agent can occupy one of
the three states which is visible to other agents. The *private
preference* of each agent is an assigned and unchanging value
representing their unspoken privately held opinion on the regime in
power, but each agent’s *public preference* is self determined in the
sense that each individual agent decides their publicly viewable state
by incorporating spatial information viewable by all agents, in
combination with their internal non-public information. The three states
or *public preferences* of Citizen agents are **Support**, **Oppose**,
and **Active**. Citizen agents can inhabit a fourth state **Jailed**
imposed on it by Security forces where they are removed from the board
and await release.

The exogenous factors within the model are the two above variables,
*private preference* and *epsilon*, as well as *vision*, *Citizen
Density*, *Security Density*, *threshold* *T*<sub>*C*</sub>, and
*maximum jail term* *J*. *Vision* refers to each agents vision radius.
Each agent has a set distance at which they can view other agents. While
this can be understood as a literal representation of spatially local
information restrictions, it also represents an abstraction of limited
information. This variable can be adjusted separately for each class of
agent, Citizen or Security, to represent more restrictions on Citizen
agents’ information access while holding Security constant, but the
model default is *vision* radius 7 for both agent classes.

Decisions on whether to change state or *public preference* is based on
the exogenous global variable *threshold* *T*<sub>*C*</sub>. Each
agent’s individual *epsilon* *ϵ*<sub>*i*</sub> interacts with the global
threshold value to set their own personal threshold for activation. The
changes in standard deviation of *epsilon* correspond to the differing
regime types, with more information-controlled societies having a lower
*epsilon*, aka lower standard deviation in a Gaussian distribution,
while lower information-controlled societies have a higher *epsilon*, or
a higher standard deviation in a Gaussian distribution.

*Citizen Density* and *Security Density* determines the number of
Citizen and Security agents as a percentage of available space within
the spatial grid. Using Epstein’s Civil Violence Model as a starting
point for understanding the behavior of different agent densities, the
*Citizen Density* was set at 0.7 and *Security Density* was allowed to
fluctuate between 0.00 and 0.09. *Maximum Jail Term* *J* is the maximum
integer value that a Security agent can impose on an active Citizen
agent. This is applied stochastically during an arrest as a uniform
distribution between 0 and *Maximum Jail Term*.

The model works on a multilevel grid where multiple agents can occupy
the same grid square simultaneously by default. Through the activation
of a user parameter the model can operate on a single layer grid where
each grid cell is limited to a single agent at a time. The grid itself
either in single-layer or multi-layer is a torus where the top, bottom,
and sides are connected. Agent vision and movement is able to jump from
the bottom to the top, or from one side to another. The grid has a
height of 40 squares, and a width of 40 squares for 1,600 total squares.
This grid size also defines the agent count via the *Citizen Density*
and *Security Density* variables expressed as a proportion of total
squares on the grid. Agents move one square per step in their Moore
neighborhood if any move is available. A Moore neighborhood includes all
squares adjacent to a given square both orthogonally and diagonally, for
a total of eight neighbors in a two-dimensional grid, excluding the
square the agent inhabits. An agent’s spatial vision is defined as the
radius of the exogenous *vision* variable in their Moore neighborhood.
Thus, an agent with *vision* radius 7, the default value in the model,
would be assessing (2\**v*+1)<sup>2</sup> = (2\*7+1)<sup>2</sup> = 225
squares in their vicinity, some of which will contain a single agent,
some multiple agents, and some no agents in the default multi-layer
setup.

The model is split into two temporal stages for each agent’s decision
and action phase using a simultaneous activation scheduler. The
scheduler first loops through each agent in a random sequence where
agents decide on their future state in a static environment. The
scheduler then loops through the agents a second time in a random
sequence who then activate their stored chosen state or *public
preference* and then take their actions. With this temporal setup, all
states of each individual agent in the step function are predetermined
during the first loop in a static environment, and so state declarations
by any agent are independent of the evolving state declarations of other
agents in the action step.

## Further Reading

This model uses inspiration from:

[Kuran, Timur. "Now Out of Never: The Element of Surprise in the East European Revolutions of 1989." World Politics, Vol. 44 No. 1, Oct. 1991: 7-48.](https://pdodds.w3.uvm.edu/files/papers/others/1991/kuran1991.pdf)

[Epstein, J. “Modeling civil violence: An agent-based computational approach”, Proceedings of the National Academy of Sciences, Vol. 99, Suppl. 3, May 14, 2002](http://www.pnas.org/content/99/suppl.3/7243.short)

[Wolf Sheep Predation model](https://github.com/projectmesa/mesa-examples/tree/main/examples/wolf_sheep)
