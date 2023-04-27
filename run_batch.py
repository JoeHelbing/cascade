import logging as log
import os
from mpi4py import MPI
from datetime import datetime
import time


start_time = time.time()

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# set up logging to output to cwd /data
# log debug messages to file
# info message to console
cwd = os.getcwd()
log_path = os.path.join(cwd, "./log/")
os.makedirs(log_path, exist_ok=True)

data_path = os.path.join(cwd, "./data/")
os.makedirs(data_path, exist_ok=True)

# Get the current date as a string in the format 'YYYY-MM-DD'
current_date = datetime.now().strftime("%Y-%m-%d")

# Create a date folder in the data directory
date_data_path = os.path.join(data_path, current_date)
os.makedirs(date_data_path, exist_ok=True)

# Create model and agent folders inside the date folder
os.makedirs(os.path.join(date_data_path, "model/"), exist_ok=True)
os.makedirs(os.path.join(date_data_path, "agent/"), exist_ok=True)
os.makedirs(os.path.join(date_data_path, "model_end/"), exist_ok=True)


from resistance_cascade.model import ResistanceCascade
from mesa.batchrunner import FixedBatchRunner
import pandas as pd
from itertools import product
from resistance_cascade.agent import Citizen, Security

log.basicConfig(filename=f"{cwd}/log/batch.log", level=log.DEBUG)

# parameters that will remain constant
fixed_parameters = {"multiple_agents_per_cell": True, "threshold": 2.94444}

# params = {
#     "seed": [*range(344000, 344500)],
#     "private_preference_distribution_mean": [-1, -0.8, -0.5],
#     "security_density": [0.01, 0.02, 0.03, 0.04],
#     "epsilon": [0.1, 0.2, 0.5, 0.8, 1],
# }

# params = {
#     "seed": [*range(344000, 344100)],
#     "private_preference_distribution_mean": [-1, -0.8, -0.5, -0.2, 0],
#     "security_density": [0.01, 0.02, 0.03, 0.04],
#     "epsilon": [0.1, 0.2, 0.5, 0.8, 1],
# }

params = {
    "seed": [344031],
    "private_preference_distribution_mean": [-0.5],
    "security_density": [0.02],
    "epsilon": [0.2, 1],
    "threshold": [2.94444],
}

# params = {
#     "seed": [*range(344000, 344500)],
#     "private_preference_distribution_mean": [-0.9, -0.7, -0.6, -1, -0.8, -0.5],
#     "security_density": [0.01, 0.02, 0.03, 0.04, 0.05],
#     "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5],
# }

# MidwayCluster Test
# params = {
#     "seed": [*range(5500, 5520)],
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
    "actives_in_vision": "actives_in_vision",
    "opposed_in_vision": "opposed_in_vision",
    "support_in_vision": "support_in_vision",
    "security_in_vision": "security_in_vision",
    "perception": "perception",
    "arrest_prob": "arrest_prob",
    "active_level": "active_level",
    "oppose_level": "oppose_level",
    "flip": "flip",
    "ever_flipped": "ever_flipped",
    "model_seed": "dc_seed",
    "model_security_density": "dc_security_density",
    "model_private_preference": "dc_private_preference",
    "model_epsilon": "dc_epsilon",
    "model_threshold": "dc_threshold",
}

# Generate the list of all possible parameter combinations
all_parameters_list = list(dict_product(params))

folder_path = "data/2023-04-25/model/"

filenames = os.listdir(folder_path)

# Filter out parameter combinations that already have a corresponding file
def file_exists(param_dict):
    filename = f"model_seed_{param_dict['seed']}_pp_{param_dict['private_preference_distribution_mean']}_sd{param_dict['security_density']}_ep_{param_dict['epsilon']}_th{fixed_parameters['threshold']}.parquet"
    return os.path.exists(os.path.join(folder_path, filename))

filtered_parameters_list = [param_dict for param_dict in all_parameters_list if not file_exists(param_dict)]

# Divide the parameter list into blocks of 20
block_size = 10
parameter_blocks = chunks(filtered_parameters_list, block_size)
max_steps = 500
# Initialize the dynamic load balancing
if rank == 0:  # If it's the master rank
    log.info("Starting batch run")
    print(f"Number of parameter combinations: {len(filtered_parameters_list)}")
    log.info(f"Number of parameter combinations: {len(filtered_parameters_list)}")
    print(f"Number of blocks: {len(parameter_blocks)}")
    log.info(f"Number of blocks: {len(parameter_blocks)}")

    next_block_index = 0
    received_blocks = 0  # Add a counter for received blocks

    for i in range(1, size):  # Assign an initial block to each worker rank
        if next_block_index < len(parameter_blocks):
            comm.send(
                (parameter_blocks[next_block_index], next_block_index), dest=i, tag=100
            )
            next_block_index += 1
        else:
            comm.send(("DONE", -1), dest=i, tag=100)

    # Receive the results from the worker ranks and write them to CSV files
    while received_blocks < len(parameter_blocks):  # Change the loop condition
        data = comm.recv(source=MPI.ANY_SOURCE, tag=200)
        (
            rank_sender,
            block_num,
            # batch_end_model,
            batch_step_model_raw,
            batch_step_agent_raw,
        ) = data

        print(f"Received block {block_num} from rank {rank_sender}")
        log.info(f"Received block {block_num} from rank {rank_sender}")

        # batch_end_model.to_parquet(
        #     f"{date_data_path}/model_end/model_block_{block_num}_rank_{rank_sender}.parquet"
        # )

        for key, df in batch_step_agent_raw.items():
            df.to_parquet(
                f"{date_data_path}/agent/agent_seed_{key[0]}_pp_{key[1]}_sd{key[2]}_ep_{key[3]}_th{key[4]}.parquet"
            )

        received_blocks += 1  # Increment the received blocks counter

        # Send a new block to the worker rank that just finished
        if next_block_index < len(parameter_blocks):
            comm.send(
                (parameter_blocks[next_block_index], next_block_index),
                dest=rank_sender,
                tag=100,
            )
            next_block_index += 1
            print(f"Sent block {next_block_index} to rank {rank_sender}")
            log.info(f"Sent block {next_block_index} to rank {rank_sender}")
        else:
            comm.send(("DONE", -1), dest=rank_sender, tag=100)
        
        for key, df in batch_step_model_raw.items():
            df.to_parquet(
                f"{date_data_path}/model/model_seed_{key[0]}_pp_{key[1]}_sd{key[2]}_ep_{key[3]}_th{key[4]}.parquet"
            )



else:  # If it's a worker rank
    while True:
        # Receive a block of parameters from the master rank
        block, block_num = comm.recv(source=0, tag=100)

        if block == "DONE":  # If there are no more blocks, exit the loop
            break

        parameters_list = block
        batch_run = FixedBatchRunner(
            ResistanceCascade,
            parameters_list,
            fixed_parameters,
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            max_steps=max_steps,
        )

        batch_run.run_all()

        # batch_end_model = batch_run.get_model_vars_dataframe()
        batch_step_model_raw = batch_run.get_collector_model()
        batch_step_agent_raw = batch_run.get_collector_agents()

        # Send the results back to the master rank for writing
        comm.send(
            (
                rank,
                block_num,
                # batch_end_model,
                batch_step_model_raw,
                batch_step_agent_raw,
            ),
            dest=0,
            tag=200,
        )

# When all blocks have been processed, the master rank sends a "DONE" message to each worker rank
if rank == 0:
    for i in range(1, size):
        comm.send(("DONE", -1), dest=i, tag=100)

end_time = time.time()
time_taken = end_time - start_time
print(f"Job completed in {time_taken} seconds.")
log.info(f"Job completed in {time_taken} seconds.")
