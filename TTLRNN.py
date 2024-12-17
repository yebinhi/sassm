import torch
from torch import nn


class Task(nn.Module):
    def __init__(self, dim):
        super(Task, self).__init__()
        self.dim = dim
        self.t_k = nn.Parameter(torch.randn(1, dim, dim))
        self.t_v = nn.Parameter(torch.randn(1, dim, dim))
        self.t_q = nn.Parameter(torch.randn(1, dim, dim))

    def loss(self, f, x):
        train_view = x @ self.t_k
        label_view = x @ self.t_v
        return nn.functional.mse_loss(f(train_view), label_view)

class Learner(nn.Module):
    def __init__(self, task: Task, dim):
        super(Learner, self).__init__()
        self.task = task
        self.model = nn.Linear(dim, dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_step(self, x):
        self.optim.zero_grad()
        loss = self.task.loss(self.model, x)
        loss.backward(retain_graph=True)
        self.optim.step()

    def predict(self, x):
        view = x @ self.task.t_q
        return self.model(view)

class TTLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TTLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = Task(input_dim)
        self.learner = Learner(self.task, input_dim)

    def forward(self, in_seq):
        self.learner.train_step(in_seq)
        return self.learner.predict(in_seq)


class CustomRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1, in_channels=3, out_channels=3, cnn_kernel_size_l=500, cnn_kernel_size_s=10):
        super(CustomRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the weights and biases for the RNN cell
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size*3))  # Weight for input to hidden
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))  # Weight for hidden to hidden
        self.b_h = nn.Parameter(torch.randn(hidden_size))  # Bias for hidden state

        # Define the weights and biases for the output layer
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))  # Weight for hidden to output
        self.b_o = nn.Parameter(torch.randn(hidden_size))  # Bias for output
        self.conv1d_l = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=cnn_kernel_size_l,
                                  stride=1, padding=0)
        self.conv1d_s = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=cnn_kernel_size_s,
                                  stride=1, padding=0)

        self.ttl = TTLinear(hidden_size, hidden_size)

        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input1, input2, input3, h):
        # CNN part for input1
        input1 = input1.permute(0, 2, 1)
        input2 = input2.permute(0, 2, 1)
        input3 = input3.permute(0, 2, 1)
        x_l = self.conv1d_l(input1)  # Change the input shape to (batch_size, channels, length)
        x_s = self.conv1d_s(input2)
        # Concatenate with input2
        x = torch.cat((x_l, x_s, input3), dim=1)  # Add sequence dimension
        x = x.permute(0, 2, 1)
        # Initialize a list to collect the outputs at each time step
        outputs = []

        # Iterate over each time step
        for t in range(x.size(1)):
            # Extract the input at time step t
            x_t = x[:, t, :]
            # input = x_t @ self.W_ih.T + h @ self.W_hh.T + self.b_h
            # Update the hidden state
            # h = torch.tanh(x_t @ self.W_ih.T + h @ self.W_hh.T + self.b_h)
            # tmp = x_t @ self.W_ih.T + h @ self.W_hh.T + self.b_h
            # t1 = x_t @ self.W_ih.T
            # print('-------1------------')
            # print(t1.shape)
            # t2 = h @ self.W_hh.T
            # print('-------2------------')
            # print(t2.shape)
            # t3 = self.b_h
            # print('-------3------------')
            # print(t3.shape)
            # t4 = t1+t2+t3
            out = (x_t @ self.W_ih.T) + (h @ self.W_hh.T) + self.b_h
            hi = self.ttl(out).squeeze(dim=0)

            # Compute the output
            y = hi @ self.W_ho.T + self.b_o

            # Store the output
            outputs.append(y)

        # Stack outputs across time steps and return the final output
        outputs = torch.stack(outputs, dim=1)
        outputs = self.out_layer(outputs[:, -1, :])
        return outputs, h

    def init_hidden(self, batch_size):
        # Initialize the hidden state with zeros
        return torch.zeros(batch_size, self.hidden_size)


# # Define input parameters
# batch_size = 32
# seq_len = 50
# input_dim = 3
# d_model = 32
# output_dim = 1
# input_size = 3
# hidden_size = 64
# output_size = 1
#
#
# input = torch.randn(batch_size, seq_len, input_size)
#
# model = CustomRNN(input_size=input_size, hidden_size = hidden_size, output_size = output_size)
#
# h = model.init_hidden(batch_size)
# # Perform a forward pass
# output, h= model(input, h)
# print(output.shape)
# print(h.shape)