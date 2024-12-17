import os
import time

from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from LSTM import LSTMModel
from SlidingWindowDataset import SlidingWindowDataset
import matplotlib
import matplotlib.pyplot as plt
from FeedForwardNN import FeedforwardNN

from TTL import TTLMLP, TTLTransformerBlock, TwinTTL

torch.autograd.set_detect_anomaly(True)


matplotlib.use('Agg')  # Use a non-interactive backend


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, result_folder):
    epoch_losses = []
    model.train()
    print("-------------------------------start training--------------------------------------")
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()  # Start time for the epoch
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Split inputs into input1 and input2
            # input1_l = inputs[:, :-1, :]  # Shape: (batch_size, sequence_length - 1, input_size)
            # input_m = inputs[:, -250:, :]
            # input_s = inputs[:, -50:, :]
            # input_cur = inputs[:, -1:, :]  # Shape: (batch_size, 1, input_size)

            optimizer.zero_grad()
            outputs = model(inputs)

            if torch.isnan(outputs).any():
                print(f"NaN detected in outputs at epoch {epoch}")
            loss = criterion(outputs.squeeze(), targets)
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_duration = time.time() - start_time  # End time for the epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f} seconds, Learning Rate: {current_lr:.6f}")

        scheduler.step()

    print("Training completed.")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = f"{model_save_path}_LSTM_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

    # Plot the loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    # Save the plot as a JPEG file
    plot_save_path = os.path.join(result_folder, f'training_loss_LSTM_{timestamp}.png')
    plt.savefig(plot_save_path, format='png')
    plt.close()
    print(f"Plot saved to {plot_save_path}")

    return model_save_path


def test(dataloader_test, model, device, model_save_path, result_folder):
    print(f"------------------------Start testing------------------------------")
    # model = TTLMLP(d_model=d_model, input_dim=3, output_dim=1)
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    all_output = []
    all_target = []
    running_loss_test = 0.0
    for inputs, targets in dataloader_test:
        inputs, targets = inputs.to(device), targets.to(device)
        # Split inputs into input1 and input2
        # input1_l = inputs[:, :-1, :]  # Shape: (batch_size, sequence_length - 1, input_size)
        # input_m = inputs[:, -250:, :]
        # input_s = inputs[:, -50:, :]
        # input_cur = inputs[:, -1:, :]  # Shape: (batch_size, 1, input_size)

        optimizer.zero_grad()
        outputs = model(inputs)
        all_output.extend(outputs.cpu().detach().numpy().flatten())
        all_target.extend(targets.cpu().detach().numpy().flatten())

    rmse = calculate_rmse(all_output, all_target)
    print(f'RMSE: {rmse:.4f}')

    num_samples = len(all_output)  # Total samples in the dataset

    # Plotting
    plt.figure()
    plt.plot(range(1, num_samples + 1), all_output, label='Test result')
    plt.plot(range(1, num_samples + 1), all_target, label='True result')
    plt.xlabel('Number')
    plt.ylabel('SOC')
    plt.title(f'True and Test Results {rmse}')
    plt.legend()
    plt.grid(True)

    # Save the plot as a JPEG file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_save_path = os.path.join(result_folder, f'test_result_LSTM_{timestamp}.png')
    plt.savefig(plot_save_path, format='png')
    plt.close()
    print(f"Plot saved to {plot_save_path}")


def calculate_rmse(pred, target):
    all_output = np.array(pred)
    all_target = np.array(target)
    # Calculate Mean Squared Error
    mse = np.mean((all_output - all_target) ** 2)

    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mse)

    return rmse


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Battery soc estimator based on transformer')

    # Parameters
    model_save_path = "model/"
    csv_file_train = 'data/m_train.csv'
    csv_file_test = 'data/m_test.csv'
    result_folder = "results/"
    window_size = 1
    step_size = 1
    batch_size = 1024
    seq_l = 500
    num_epochs = 100
    initial_lr = 0.001
    gamma = 0.9
    step_size_lr = 2
    d_model = 64
    input_dim = 3
    output_dim = 1
    input_size = 3      # Number of features (current, voltage, temperature)
    hidden_size = 50    # Number of hidden units
    num_layers = 2      # Number of LSTM layers
    output_size = 1     # Number of output units (SOC)

    # define dataset
    dataset_train = SlidingWindowDataset(csv_file_train, window_size, step_size)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    dataset_test = SlidingWindowDataset(csv_file_test, window_size, step_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The program will run on: {device}')

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=step_size_lr, gamma=gamma)

    model_save_path = train(model, dataloader_train, criterion, optimizer, scheduler, num_epochs, device,
                            model_save_path, result_folder)

    test(dataloader_test, model, device, model_save_path, result_folder)
    # test(dataloader_test, model, device, "model/green_transformer_model_20240807_195308.pth", result_folder)
