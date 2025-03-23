import numpy as np
import torch
import scipy.sparse as sp
from typing import Dict
from spikenet import neuron, dataset
from helper import save_test_results


def get_DADx(adj, x, a=0.5, b=0.5):
  degree = np.array(adj.sum(1)).flatten()
  D_inv_a = np.power(degree, -a, where=degree!=0)
  D_inv_b = np.power(degree, -b, where=degree!=0)
  D_inv_a = sp.diags(D_inv_a)
  D_inv_b = sp.diags(D_inv_b)
  transformed_x = D_inv_a @ adj @ D_inv_b @ x
  return torch.FloatTensor(transformed_x)

def _generate_dynamic_spike_train(data: dataset.Dataset, hp: Dict, snn) -> torch.Tensor:
    """
    data.adj : shape = (T, N, N) or list of length T of adjacency matrices
    data.x   : shape = (T, N, F) or list of length T of node features
    T = number of snapshots
    N = number of nodes
    F = number of node features

    hp["time_steps"] = how many internal SNN time steps to simulate per snapshot
    hp["a"], hp["b"] = exponents in D^(-a) A D^(-b)
    """
    spike_train_all = []
    DADx_prev = None
    spikes_prev = None
    threshold = hp["threshold"]

    total_nodes = 0
    saved_nodes = 0
    total_spikes = 0
    saved_spikes = 0

    T = len(data.adj)       # number of snapshots
    # 2) Loop over each snapshot
    for t in range(T):
        if "dynamic_reset" not in hp or hp["dynamic_reset"]:
          snn.reset()
        adj_t = data.adj[t]  # NxN
        x_t = data.x[t]      # NxF
        DADx_t = get_DADx(adj_t, x_t, a=hp["a"], b=hp["b"])

        if DADx_prev is not None:
            # delta shape: (num_nodes,), mask shape: (num_nodes,)
            delta = torch.abs(DADx_t - DADx_prev).max(dim=1)[0] # Get the max feature difference for each node
            mask = delta < threshold # Mask nodes whose change is insignificant
            # Count the number of nodes whose change is insignificant
            total_nodes += mask.size(0)
            saved_nodes += mask.sum().item()
            DADx_t[mask] = 0

        spike_trains_this_snapshot = []
        for _ in range(hp["time_steps"]):
            spikes = snn(DADx_t)
            spike_trains_this_snapshot.append(spikes)
        spikes_t = torch.stack(spike_trains_this_snapshot)

        if spikes_prev is not None:
            for i in range(hp["time_steps"]):
                x = spikes_prev[i][mask]
                saved_spikes += x.sum().item()
                spikes_t[i][mask] = spikes_prev[i][mask]
        
        total_spikes += spikes_t.sum().item()
        spike_train_all.append(spikes_t)
        DADx_prev = DADx_t
        spikes_prev = spikes_t

    # 3) Concatenate all T snapshots if you’d like: shape = (T, time_steps, N, F)
    spike_train_all = torch.stack(spike_train_all, dim=0)
    spike_train_all = spike_train_all.view(-1, spike_train_all.size(-2), spike_train_all.size(-1))

    # (Optional) convert to bool for memory savings
    spike_train_all = spike_train_all.to(torch.bool)

    save_test_results(hp["hyperparameters_id"], {"Total Nodes": total_nodes, "Saved Nodes": saved_nodes, "Total Spikes": total_spikes, "Saved Spikes": saved_spikes})

    return spike_train_all

def _generate_static_spike_train(data: dataset.Dataset, hp: Dict, snn) -> torch.Tensor:
  spike_train = []
  DADx = get_DADx(data.adj[-1], data.x[-1], a=hp["a"], b=hp["b"])
  for _ in range(hp["time_steps"]):
    spike_train.append(snn(DADx))
  return torch.stack(spike_train).to(torch.bool)

def generate_spike_train(data: dataset.Dataset, hp: Dict) -> torch.Tensor:
  if hp["act"] == "IF":
      snn = neuron.IF(alpha=hp["alpha"], surrogate=hp["surrogate"])
  elif hp["act"] == "LIF":
      snn = neuron.LIF(tau=hp["tau"], alpha=hp["alpha"], surrogate=hp["surrogate"])
  elif hp["act"] == "PLIF":
      snn = neuron.PLIF(tau=hp["tau"], alpha=hp["alpha"], surrogate=hp["surrogate"])

  if hp["graph_type"]=="static":
    return _generate_static_spike_train(data, hp, snn)
  else:
    # Final shape should be (time_steps, num_nodes, num_features)
    return _generate_dynamic_spike_train(data, hp, snn)
  
def prune_spikes(spike_train, hp: Dict) -> torch.Tensor:
  num_spikes = torch.sum(spike_train, dim=(1,2))
  prune_param = hp["prune_param"]
  median = torch.median(num_spikes)
  pruned_start_idx = 0
  while(num_spikes[pruned_start_idx] < median * prune_param):
    pruned_start_idx += 1
  return spike_train[pruned_start_idx:]