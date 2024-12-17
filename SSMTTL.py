import numpy as np
import torch
import torch.nn as nn


class DiscreteStateSpaceModel(nn.Module):
    def __init__(self, h_dim, input_dim, output_dim, T, device):
        super(DiscreteStateSpaceModel, self).__init__()

        self.device = device
        self.h_dim = h_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T

        self.input_linear = nn.Linear(input_dim, h_dim).to(device)

        self.B = nn.Parameter(torch.randn(h_dim, h_dim, device=device))
        self.C = nn.Parameter(torch.randn(h_dim, h_dim, device=device))
        self.output_linear = nn.Linear(h_dim, output_dim).to(device)

        with torch.no_grad():
            self.A = self.hippo_matrix(h_dim).to(device)
            self.A_d = torch.matrix_exp(self.T * self.A)
            self.B_d = torch.linalg.pinv(self.A) @ (self.A_d - torch.eye(self.A.size(0), device=self.device)) @ self.B

    def forward(self, u_t):
        u_t = self.input_linear(u_t)

        if self.training:
            return self.global_convolution(u_t)
        else:
            return self.linear_recurrence(u_t)

    def hippo_matrix(self, n):
        # Step 1: Generate the HiPPO matrix using NumPy
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    M[i, j] = 2 * i + 1
                elif i > j:
                    M[i, j] = np.sqrt(2 * i + 1) * np.sqrt(2 * j + 1)

        # Step 2: Convert the NumPy array to a PyTorch tensor
        M_tensor = torch.tensor(M, dtype=torch.float32)

        return M_tensor

    def global_convolution(self, u):
        batch_size, seq_length, input_dim = u.shape

        if batch_size < 1:
            raise ValueError("Batch size for testing data should be greater than 1.")

        k = self.compute_kernel(seq_length)
        outputs = torch.einsum('nij,ijk->nik', u, k)
        out = self.output_linear(outputs[:, -1, :])
        return out

    def compute_kernel(self, seq_length):
        # Compute the kernel K based on the SSM parameters
        # K = [CB, CAB, CA^2B, ..., CA^kB]
        K = [self.C @ (torch.matrix_power(self.A_d, i) @ self.B_d) for i in range(seq_length)]
        K = torch.stack(K, dim=0)  # Shape (L, N)
        return K

    def compute_discrete_A(self):

        return torch.matrix_exp(self.T * self.A)

    def compute_discrete_B(self):
        A_d = self.compute_discrete_A()
        return torch.linalg.pinv(self.A) @ (A_d - torch.eye(self.A.size(0), device=self.device)) @ self.B

    def linear_recurrence(self, u):
        batch_size, seq_length, input_dim = u.shape

        if batch_size > 1:
            raise ValueError('batch_size cannot be greater than 1')

        u = u.squeeze(0).to(self.device)
        h = torch.zeros(self.h_dim, device=self.device)
        for t in range(seq_length):
            x = u[t, :]
            h = torch.matmul(self.A_d, h) + torch.matmul(self.B_d, x)
            y = torch.matmul(self.C, h)

        out = self.output_linear(y)
        return out



# # Example usage
# state_dim = 64
# input_dim = 3
# output_dim = 1
# T = 0.1  # Sampling period
# batchsize = 2048
# seq_l = 501
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Initialize the discrete state space model
# model = DiscreteStateSpaceModel(state_dim, input_dim, output_dim, T, device)
# model.to(device)
#
# # # Define input
# # u_t = torch.randn(batchsize, seq_l, input_dim).to(device)
# # # print('-------------------------------------------------')
# # model.train()
# # output = model(u_t)
# # print(output.shape)
#
# # print('-------------------------------------------------')
# # model.eval()
# # output2 = model(u_t)
# # print(output2.shape)
#
# # Parameters
# num_samples = 1000
# num_features = 10
# batch_size = 32
# shuffle = True
# num_workers = 1  # Number of subprocesses to use for data loading
#
# # Instantiate the dataset
# dataset = RandomDataset(num_samples, num_features)
#
# # Instantiate the DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=shuffle,
#     num_workers=num_workers
# )
#
# # Example usage
# for batch in dataloader:
#     features, labels = batch
#     print(f'Features: {features.shape}, Labels: {labels.shape}')
