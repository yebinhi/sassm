import math

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from typing import List
import seaborn as sns

class Task(nn.Module):
    def __init__(self, dim):
        super(Task, self).__init__()
        self.dim = dim
        self.t_k = nn.Parameter(torch.randn(1, dim, dim))
        self.t_v = nn.Parameter(torch.randn(1, dim, dim))
        self.t_q = nn.Parameter(torch.randn(1, dim, dim))

    def loss(self, f, x: Tensor) -> Tensor:
        train_view = x @ self.t_k
        label_view = x @ self.t_v
        return nn.functional.mse_loss(f(train_view), label_view)

class Learner(nn.Module):
    def __init__(self, task: Task, dim):
        super(Learner, self).__init__()
        self.task = task
        self.model = nn.Linear(dim, dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_step(self, x: Tensor):
        self.optim.zero_grad()
        loss = self.task.loss(self.model, x)
        loss.backward(retain_graph=True)
        self.optim.step()

    def predict(self, x: Tensor) -> Tensor:
        view = x @ self.task.t_q
        return self.model(view)

class TTLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TTLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = Task(input_dim)
        self.learner = Learner(self.task, input_dim)

    def forward(self, in_seq: Tensor):
        self.learner.train_step(in_seq)
        return self.learner.predict(in_seq)

class TTLMLP(nn.Module):
    def __init__(self, d_model: int, input_dim, output_dim):
        super(TTLMLP, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.fc1 = TTLinear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()
        self.fc3 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_linear(x)
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x + residual)
        x = self.fc3(x[:, -1, :])
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        encoding = self.encoding.to(x.device)
        return x + encoding[:, :x.size(1)]


class TTLTransformerBlock(nn.Module):
    def __init__(self, d_model, input_dim, output_dim, seq, dropout_rate=0.1):
        super(TTLTransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.seq = seq

        self.q_linear = nn.Linear(input_dim, d_model)
        self.k_linear = nn.Linear(input_dim, d_model)
        self.v_linear = nn.Linear(input_dim, d_model)

        self.ttl_linear = TTLinear(d_model, output_dim)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(d_model * seq, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(0, 2, 1)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Apply dropout after linear transformations
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        out = torch.cat([q, k, v])
        out = self.ttl_linear(out)
        normed = self.norm(out)
        out = self.act(normed)

        # Apply dropout after activation
        out = self.dropout(out)

        out = out[(out.size(0)) * 2 // 3:]
        out = out.view(out.size(0), -1)

        # Apply dropout after the output layer
        out = self.out(out)

        return out


class TTLTransformer(nn.Module):
    def __init__(self, d_model, input_dim, output_dim, seq, dropout_rate=0.1):
        super(TTLTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.seq = seq

        self.q_linear = nn.Linear(input_dim, d_model)
        self.k_linear = nn.Linear(input_dim, d_model)
        self.v_linear = nn.Linear(input_dim, d_model)

        self.ttl_linear = TTLinear(d_model, output_dim)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(d_model * seq, output_dim)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q_out = self.ttl_linear(q)
        k_out = self.ttl_linear(k)
        k_out = k_out.permute(0, 2, 1)
        out = q_out * k_out
        normed = self.norm(out)
        out = self.act(normed)

        # Apply dropout after activation
        out = self.dropout(out)

        out = out[(out.size(0)) * 2 // 3:]
        out = out.view(out.size(0), -1)

        # Apply dropout after the output layer
        out = self.out(out)


class TwinTTL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, cnn_kernel_size_l=500, cnn_kernel_size_s=50, num_layers=2, d_model=64, output_size=1):
        super(TwinTTL, self).__init__()
        # Define the layers for the CNN part
        self.num_layers = num_layers
        self.d_model = d_model
        self.in_channels = in_channels
        self.cnn_kernel_size_l = cnn_kernel_size_l
        self.cnn_kernel_size_s = cnn_kernel_size_s
        self.output_size = output_size
        # Define positional encoding
        self.positional_encoding = PositionalEncoding(embed_size=d_model, max_len=500)

        self.conv1d_l = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=cnn_kernel_size_l, stride=1, padding=0)
        self.conv1d_m = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=250,
                                  stride=1, padding=0)
        self.conv1d_s = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=cnn_kernel_size_s,
                                stride=1, padding=0)

        self.fc1 = nn.Linear(12, d_model)

        # Define the layers for the RNN part
        self.ttl = TTLTransformerBlock(d_model=d_model, input_dim=d_model, output_dim=1, seq=1)

        # Define the final dense layer to output a single number
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, input1, input2, input3, input4):
        # CNN part for input1
        input1 = input1.permute(0, 2, 1)
        input2 = input2.permute(0, 2, 1)
        input3 = input3.permute(0, 2, 1)
        input4 = input4.permute(0, 2, 1)

        x_l = self.conv1d_l(input1)  # Change the input shape to (batch_size, channels, length)
        x_m = self.conv1d_m(input2)
        x_s = self.conv1d_s(input3)
        # Concatenate with input2
        concatenated = torch.cat((x_l, x_m, x_s, input4), dim=1)  # Add sequence dimension
        concatenated = concatenated.permute(0, 2, 1)

        concatenated = self.fc1(concatenated)
        # Apply positional encoding
        concatenated = self.positional_encoding(concatenated)
        # self.visualisePositionalencoding(concatenated)

        # ttl part
        out = self.ttl(concatenated)

        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        return out


    def visualisePositionalencoding(self, pos_encoding):
        # Step 3: Visualize the Positional Encoding using a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pos_encoding.squeeze(1).T.cpu().detach().numpy(), cmap='viridis') # outputs.cpu().detach().numpy()
        plt.xlabel('Sequence Position')
        plt.ylabel('Embedding Dimensions')
        plt.title('Positional Encoding')
        plt.savefig("results/positional_encoding.png", format='png')
        plt.show()





# # Define input parameters
# batch_size = 32
# seq_len = 500
# input_dim = 3
# d_model = 32
# output_dim = 1
#
#
# # Create random input tensor
# input1 = torch.randn(batch_size, 500, input_dim)
# input2 = torch.randn(batch_size, 10, input_dim)
# input3 = torch.randn(batch_size, 1, input_dim)
#
# # Initialize the TTLMLP model
# ttl = TTLTransformerBlock(d_model=d_model, input_dim = input_dim, output_dim = output_dim, seq= 500)
#
# # Perform a forward pass
# output = ttl(input1)
#
# # Print the shapes of the input and output
# print("Output shape:", output.shape)