#!/bin/bash

# Create a new environment using mamba and the environment.yml file
mamba env create -f environment.yml

# Verify the installation
mamba env list