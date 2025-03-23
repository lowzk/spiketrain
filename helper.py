from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def check_hyperparameters(hyperparameters):
  if hyperparameters['dataset'] not in ['DBLP',]:
    raise Exception("Invalid dataset name")
  if (hyperparameters['a']+hyperparameters['b']!=1):
    raise Exception("a+b must be equal to 1")
  if hyperparameters['a']<0 or hyperparameters['b']<0:
    raise Exception("a and b must be positive")
  if hyperparameters["graph_type"] not in ["static", "dynamic"]:
    raise Exception("Invalid graph type, only static and dynamic are allowed")
  if hyperparameters["graph_type"]=="static":
    if hyperparameters["time_steps"] is None:
      raise Exception("time_steps is required for static graph")
  if hyperparameters["act"] not in ["IF", "LIF", "PLIF"]:
    raise Exception("Invalid activation function, only IF, LIF and PLIF are allowed")
  
def print_confusion_matrix(all_labels, all_preds):
  cm = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
              xticklabels=range(10), yticklabels=range(10))
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()

def get_tensor_memory(tensor):
  element_size = tensor.element_size()  # Size of each element in bytes
  num_elements = tensor.numel()         # Total number of elements
  total_memory = element_size * num_elements      # Total memory in bytes
  total_memory_mb = total_memory / (1024 ** 2)     # Convert to megabytes
  return total_memory_mb

def save_test_results(name, metrics):
  with open(f"test_results/{name}_results.txt", "a") as f:
    for key, value in metrics.items():
      f.write(f"{key}: {value}\n")

def read_test_results(name):
  with open(f"test_results/{name}_results.txt", "r") as f:
    lines = f.readlines()
  metrics = {}
  for line in lines:
    key, value = line.strip().split(": ")
    metrics[key] = float(value)
  return metrics

def reset_test_results(name):
  with open(f"test_results/{name}_results.txt", "w") as f:
    f.write("")

def display_dicts_as_table(dicts, names):
    if not dicts or not names or len(dicts) != len(names):
        print("Invalid input: Ensure dictionaries and names have the same length.")
        return
    
    # Collect all possible keys
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    all_keys = sorted(all_keys)  # Sort keys for consistency
    
    # Construct table data
    table_data = []
    for name, d in zip(names, dicts):
        row = [name] + [d.get(key, "NIL") for key in all_keys]
        table_data.append(row)
    
    # Create table headers
    headers = ["Name"] + list(all_keys)
    
    # Display table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
