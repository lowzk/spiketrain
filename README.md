# SpikeTrain: A memory-efficient method for graph representation learning based on SNNs

This repository is an official PyTorch implementation of SpikeTrain done by Low Zhe Kai for his Final Year Project in the Nanyang Technological University of Singapore under SCSE.

This project was done under the guidance of Asst Prof Luo Siqiang and Zulun Zhu. 

## Basic set up

1. Download DBLP dataset at the [following link](https://www.dropbox.com/scl/fo/5w0a14icfv4o7t0azrqda/AOLp-_Kd1s_USOzeoAttE7w?rlkey=qhx7csgahlcbuppjx4ewa3l0o&e=1&dl=0) and place it in the root directory. Make sure that the folder is named `data`
2. Start a Python virtual environment and install the dependencies in `requirements.txt`
3. Run the baseline SpikeTrain program by running `python3 main.py` or `python main.py`

## Abstract
Graph Neural Networks (GNNs) have demonstrated exceptional performance across various domains, leveraging message-passing mechanisms to learn intricate relationships within graph-structured data. However, their widespread adoption is hindered by significant computational and memory overhead, particularly in dynamic graph representation learning where continuous updates exacerbate resource demands. To address this challenge, we introduce a novel spike-based framework called SpikeTrain that integrates Spiking Neural Networks (SNNs) into GNN architectures. Our approach employs a binary spike train encoding for graph data, reducing storage requirements, enhancing computational efficiency, and lowering energy consumption while preserving expressive power. Additionally, we implement a lazy-update algorithm for efficient spike generation, spike pruning for memory optimization, and tensor compression for space-efficient storage. Through extensive experimentation on the DBLP dataset, SpikeTrain is able to achieve substantial memory savings whilst retaining comparable classification accuracy. This research paves the way for scalable, resource-efficient graph learning, enabling real-time processing of large-scale, dynamic graphs with minimal computational cost. 

## Usage

The program is written such that there is a single entry point in the `get_results` function which can be found in `pipeline.py`. The inputs to this function are the hyperparameters and system parameters, which have the following formats.

### Hyperparameters
```json
{
  "hyperparameters_id": "baseline_dynamic_noreset", 
  "dataset": "DBLP",
  "graph_type": "dynamic",
  "dynamic_reset": false,
  "time_steps": 5,
  "tau": 1.0,
  "alpha": 1.0,
  "surrogate": "triangle", 
  "act": "LIF",
  "a": 0.5,
  "b": 0.5,
  "prune_param": null,
  "model": "MLP",
  "threshold": 0.5
}
```

### System parameters
```json
{
  "batch_size": 64,
  "num_epochs": 20,
  "verbose": false,
  "test_memory": true,
  "save_tensor": false,
  "test_size": 0.2
}
```

By tweaking these parameters, you can run various experiments on the dataset. A single entry point is provided in `main.py`, where the configuration can be specified as the arguments when running the program. Alternatively, easy experimentation can be conducted in `experiments.ipynb` where there is already existing tests and templates that can be used.

### Saving results

When running experiments, the `hyperparameters_id` parameter in the configuration file specifies where the experiment result will be saved, and all results will be saved in the `test_results` folder under the name `{hyperparemter_id}_results.txt`.

## Code details
- `main.py`: Main entry point to run a single training and testing cycle of SpikeTrain. Configuration files can be parsed as arguments: 
```bash
python3 main.py -hp config/baseline_dynamic.json -sp config/system_params.json
```
- `experiments.ipynb`: Contains the individual experiments ran for hyperparameter tuning and benchmarking
- `pipeline.py`: Contains the main pipeline for the entire training and testing process
- `spike_generation.py`: Contains the logic for spike generation for static and dynamic graphs
- `models.py`: Contains the downstream LSTM and MLP models for node classification
- `compressor.py`: Contains the logic for spike compression
- `helper.py`: Contains helper functions
- `config/`: Contains the configuration files for the various experiments
- `test_results/`: Contains the results for the various experiments
- `spikenet/`: Contains the dataset and spiking neuron logic, reused from SpikeNet's architecture

## Details

More details regarding the effectiveness of SpikeTrain, as well as the theoretical aspects, can be found in the final report.