import torch.nn as nn
from torch.utils.data import Dataset

class SpikeTrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # Need to typecast back into a float later
        self.y = y.long()   # Ensure labels are long tensors

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTMClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out, (hn, _) = self.lstm(x)
    out = self.fc(hn[-1])
    return out
  
class MLPClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out