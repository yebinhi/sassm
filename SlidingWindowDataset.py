import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    def __init__(self, csv_file, window_size=512, step_size=1):
        # Read the CSV file into a DataFrame
        self.data = pd.read_csv(csv_file, header=None, memory_map=True)
        self.window_size = window_size
        self.step_size = step_size
        self.num_features = self.data.shape[1] - 1  # Number of feature columns (excluding the last column)

        # Convert DataFrame to numpy array
        self.data = self.data.values
        self.n, self.m = self.data.shape
        self.num_slices = self.n - self.window_size
        self.indices = [(i, i + self.window_size) for i in range(0, self.n - self.window_size + 1, self.step_size)]

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        if idx >= self.num_slices:
            raise IndexError("Index out of range")

        start_idx, end_idx = self.indices[idx]
        # Get the window of data for the input features (excluding the last column)
        window_data = self.data[start_idx:end_idx, :-1]
        # Get the window of data for the target values (the last column)
        window_target = self.data[end_idx - 1, -1]
        return torch.tensor(window_data, dtype=torch.float32), torch.tensor(window_target, dtype=torch.float32)


# # Parameters
# csv_file = 'data/m_train.csv'  # Replace with your actual CSV file path
# window_size = 1
# step_size = 1
# batch_size = 16  # Adjust batch size as needed
# num_epochs = 10  # Replace with your desired number of epochs
#
# # Create the dataset and data loader
# dataset = SlidingWindowDataset(csv_file, window_size, step_size)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
# # Example usage in a training loop
# for epoch in range(num_epochs):
#     for inputs, targets in dataloader:
#         # Your training logic here
#         # Example: output = model(inputs)
#         print(f'Input batch shape: {inputs.shape}, Target batch shape: {targets.shape}')
