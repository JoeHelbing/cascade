import logging as log
import os

# set up logging to output to cwd /data
# log debug messages to file
# info message to console
cwd = os.getcwd()
log_path = os.path.join(cwd, "./log/")
if not os.path.exists(log_path):
    os.makedirs(log_path)

data_path = os.path.join(cwd, "./data/")
if not os.path.exists(data_path):
    os.makedirs(data_path)

from resistance_cascade.model import ResistanceCascade
from mesa.batchrunner import FixedBatchRunner
import pandas as pd
from itertools import product
from resistance_cascade.agent import Citizen, Security

log.basicConfig(filename=f"{cwd}/log/batch.log", level=log.DEBUG)
log.info("Starting batch run")

# parameters that will remain constant
fixed_parameters = {
    "multiple_agents_per_cell": True,
}

params = {
    "seed": [*range(1030, 1039)],
    "private_preference_distribution_mean": [-1, -0.8, -0.5, -0.3, 0, 0.3, 0.5],
    "security_density": [0.01, 0.02, 0.03, 0.04],
    "epsilon": [0.1, 0.2, 0.5, 0.8, 1],
}

# single block param for testing purposes
# params = {
#     "seed": [1030],
#     "private_preference_distribution_mean": [-1],
#     "security_density": [0.01],
#     "epsilon": [0.1],
# }

# Helper function to generate all possible combinations of parameters
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))


# Helper function to divide the parameter list into blocks
def chunks(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# set up the reporters
model_reporters = {
    "Seed": lambda m: m.report_seed(m),
    "Citizen Count": lambda m: m.count_citizen(m),
    "Active Count": lambda m: m.count_active(m),
    "Oppose Count": lambda m: m.count_oppose(m),
    "Support Count": lambda m: m.count_support(m),
    "Speed of Spread": lambda m: m.speed_of_spread(m),
    "Security Density": lambda m: m.report_security_density(m),
    "Private Preference": lambda m: m.report_private_preference(m),
    "Epsilon": lambda m: m.report_epsilon(m),
    "Threshold": lambda m: m.report_threshold(m),
    "Revolution": lambda m: m.report_revolution(m),
}

agent_reporters = {
    "pos": "pos",
    "condition": "condition",
    "opinion": "opinion",
    "activation": "activation",
    "private_preference": "private_preference",
    "epsilon": "epsilon",
    "oppose_threshold": "oppose_threshold",
    "active_threshold": "active_threshold",
    "jail_sentence": "jail_sentence",
    "flip": "flip",
    "ever_flipped": "ever_flipped",
    "model_security_density": "dc_security_density",
    "model_private_preference": "dc_private_preference",
    "model_epsilon": "dc_epsilon",
    "model_threshold": "dc_threshold",
    "model_seed": "dc_seed",
}

# Generate the list of all possible parameter combinations
all_parameters_list = list(dict_product(params))

# Divide the parameter list into blocks of 50
block_size = 50
parameter_blocks = chunks(all_parameters_list, block_size)

# Run each block of 50 parameter combinations
for block_num, parameters_list in enumerate(parameter_blocks):
    max_steps = 500
    batch_run = FixedBatchRunner(
        ResistanceCascade,
        parameters_list,
        fixed_parameters,
        model_reporters=model_reporters,
        agent_reporters=agent_reporters,
        max_steps=max_steps,
    )

    batch_run.run_all()

    batch_end_model = batch_run.get_model_vars_dataframe()
    batch_end_agent = batch_run.get_agent_vars_dataframe()
    batch_step_model_raw = batch_run.get_collector_model()
    batch_step_agent_raw = batch_run.get_collector_agents()

    cwd = os.getcwd()
    path = os.path.join(cwd, "data/")

    batch_end_model.to_csv(f"{path}/model_batch_{block_num}.csv")

    if not os.path.exists(f"{path}/model/"):
        os.makedirs(f"{path}/model/")
    for key, df in batch_step_model_raw.items():
        df.to_csv(
            f"{path}/model/model_seed_{key[0]}_pp_{key[1]}_sd{key[2]}_ep_{key[3]}.csv"
        )

    if not os.path.exists(f"{path}/agent/"):
        os.makedirs(f"{path}/agent/")
    for key, df in batch_step_agent_raw.items():
        df.to_csv(
            f"{path}/agent/agent_seed_{key[0]}_pp_{key[1]}_sd{key[2]}_ep_{key[3]}.csv"
        )
