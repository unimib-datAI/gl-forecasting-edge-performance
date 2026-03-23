# GL Forecasting Edge Performance

This repository contains the code and notebooks used to generate FaaS-edge 
node networks, prepare node-level datasets, and run training or Gossip 
Learning experiments for edge resource usage forecasting.

## Intended workflow

The recommended workflow is:

1. Generate a FaaS-edge node network with 
[`src/1_network_generation.ipynb`](src/1_network_generation.ipynb).
2. Prepare node datasets with 
[`src/functions_data_preparation.py`](src/functions_data_preparation.py).
3. Run experiments with either:
   - [`src/train.py`](src/train.py) for training-based experiments.
   - [`src/traingossip.py`](src/traingossip.py) for Gossip Learning experiments.

Each step depends on the previous one, so the network should be generated 
first, then the datasets prepared, and finally the desired experiment launched.

## Requirements

The project is organized around Python-based simulation and experiment 
scripts, so you should use a Python environment with the dependencies 
required by the repository (see [`src/requirements.txt`](src/requirements.txt)).

