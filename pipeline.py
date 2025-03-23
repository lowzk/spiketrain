import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile

from helper import check_hyperparameters, get_tensor_memory, print_confusion_matrix, save_test_results, reset_test_results
from spikenet import dataset
from compressor import pack_tensor, unpack_tensor
from spike_generation import generate_spike_train, prune_spikes
from models import LSTMClassifier, MLPClassifier, SpikeTrainDataset

def get_results(hyperparameters, system_params):
    check_hyperparameters(hyperparameters)

    reset_test_results(hyperparameters["hyperparameters_id"])

    # ----------------------
    # Data Loading and Setup
    # ----------------------
    if hyperparameters['dataset'] == 'DBLP':
        data = dataset.DBLP()

    original_memory = get_tensor_memory(data.x if hyperparameters["graph_type"]=="dynamic" else data.x[-1])
    original_num_elements = data.x.numel() if hyperparameters["graph_type"]=="dynamic" else data.x[-1].numel()

    spike_train = generate_spike_train(data, hyperparameters)

    if system_params["verbose"]:
      print(spike_train.shape)
    
    if hyperparameters["prune_param"] is not None:
        spike_train = prune_spikes(spike_train, hyperparameters)

    # Permute to [samples, time, features] if RNN-based model
    spike_train = spike_train.permute(1, 0, 2)

    # For MLP, flatten the [time, features] dimension
    if hyperparameters["model"]=="MLP":
        spike_train = spike_train.reshape(spike_train.shape[0], -1)  # [samples, time * features]

    if system_params["verbose"]:
      print(spike_train.shape)
    # --------------
    # Compression (packing) example
    # --------------
    compressed_spike_train, original_shape = pack_tensor(spike_train)
    # Show theoretical space savings by "packing"
    spike_train = unpack_tensor(compressed_spike_train, original_shape)
    final_memory = get_tensor_memory(compressed_spike_train)

    if system_params["save_tensor"]:
        if hyperparameters["graph_type"]=="dynamic":
            data.x.numpy().tofile(f"{hyperparameters['dataset']}_x_original.npy")
        else:
            data.x[-1].numpy().tofile(f"{hyperparameters['dataset']}_x[-1]_original.npy")
        compressed_spike_train.numpy().tofile(f"{hyperparameters['dataset']}_spike_train_compressed.npy")

    final_num_elements = spike_train.numel()
    if system_params["test_memory"] and system_params["verbose"]:
        print(f"Original memory: {original_memory:.2f} MB, Final memory: {final_memory:.2f} MB")
        print(f"Original num elements: {original_num_elements}, Final num elements: {final_num_elements}")

    save_test_results(hyperparameters["hyperparameters_id"], {"Original tensor memory": original_memory, "Final tensor memory": final_memory, "Memory savings": original_memory-final_memory})

    # -----------------------
    # Train/Test Split
    # -----------------------
    y = data.y
    X_train, X_test, y_train, y_test = train_test_split(spike_train, y,
                                                        test_size=system_params["test_size"],
                                                        random_state=42,
                                                        stratify=y)

    train_dataset = SpikeTrainDataset(X_train, y_train)
    test_dataset  = SpikeTrainDataset(X_test,  y_test)

    batch_size = system_params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ------------------------
    # Model Definition
    # ------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if system_params["verbose"]:
      print(f'Using device: {device}')

    if hyperparameters["model"] == "LSTM":
        # LSTM expects [batch_size, seq_len, input_size]
        model = LSTMClassifier(input_size=spike_train.shape[-1],
                               hidden_size=256,
                               num_layers=2,
                               num_classes=data.num_classes).to(device)
    elif hyperparameters["model"] == "MLP":
        # MLP expects [batch_size, input_size]
        model = MLPClassifier(input_size=spike_train.shape[-1],
                              hidden_size=256,
                              num_classes=data.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = system_params["num_epochs"]

    # ---------------------------------
    # Measure MACs for a Single Forward
    # ---------------------------------
    # We'll approximate training MACs as (forward MACs + backward MACs)
    # Typically backward pass ~2x forward pass => total ~3x forward pass.
    # Also measure inference MACs for the test set.

    # Get a single batch from train_loader for MAC profiling
    dummy_input, _ = next(iter(train_loader))
    dummy_input = dummy_input.float().to(device)

    # Profile the forward pass
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    if system_params["verbose"]:
      print(f"Single-batch MACs (forward): {macs:.2f}, Number of parameters: {params}")

    # Multiply by the number of training batches and epochs
    macs_per_epoch_forward = macs * len(train_loader)
    training_macs_forward = macs_per_epoch_forward * num_epochs

    # Approximate backward pass cost as 2× forward
    # (This is a rough rule of thumb, actual overhead can vary.)
    training_macs_backward = 2 * training_macs_forward

    # Total training MACs
    total_training_macs = training_macs_forward + training_macs_backward

    # Inference (test) MACs: #batches × single forward pass
    inference_macs = macs * len(test_loader)

    if system_params["verbose"]:
      print(f"Approx. total training MACs (forward+backward): {total_training_macs:.2f}")
      print(f"Approx. total test inference MACs: {inference_macs:.2f}")

    save_test_results(hyperparameters["hyperparameters_id"], {"Training MACs": total_training_macs, "Inference MACs": inference_macs})

    # --------------
    # Training Loop
    # --------------
    start_time = time.time()
    final_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)  # Convert back to float
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        if epoch+1 == num_epochs:
           save_test_results(hyperparameters["hyperparameters_id"], {"Final loss": avg_loss})

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        final_accuracy = accuracy
        if system_params["verbose"]:
            print(f'Accuracy on test set: {accuracy:.2f}%\n')

    end_time = time.time()
    time_taken = end_time - start_time

    # ----------------------------
    # Gather Predictions for CM
    # ----------------------------
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if system_params["verbose"]:
      print_confusion_matrix(all_labels, all_preds)
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')  # Macro: unweighted mean per class
    f1_micro = f1_score(all_labels, all_preds, average='micro')  # Micro: global metric
    if system_params["verbose"]:
      print(f"F1 Score (Macro): {f1_macro:.4f}, F1 Score (Micro): {f1_micro:.4f}")

    save_test_results(hyperparameters["hyperparameters_id"], {"F1 Score (Macro)": f1_macro, "F1 Score (Micro)": f1_micro})
    save_test_results(hyperparameters["hyperparameters_id"], {"Final accuracy": final_accuracy, "Time taken": time_taken})

    return final_accuracy, time_taken