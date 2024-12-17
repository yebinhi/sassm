import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from TTLRNN import CustomRNN
from TwinHybridModel import TwinHybridModel
from SlidingWindowDataset import SlidingWindowDataset
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, result_folder):
    """
    Train the RNN model.

    Parameters:
        model (nn.Module): The RNN model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to use for training (CPU or GPU).
        model_save_path (str): Path to save the trained model.
        result_folder (str): Folder to save training results (e.g., plots).

    Returns:
        str: Path to the saved model.
    """
    epoch_losses = []

    print("------------------------------- Start Training --------------------------------------")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        start_time = time.time()  # Start time for the epoch

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Split inputs into input1 and input2
            input1_l = inputs[:, :-1, :]  # Shape: (batch_size, sequence_length - 1, input_size)
            input_s = inputs[:, -10:, :]
            input_cur = inputs[:, -1:, :]  # Shape: (batch_size, 1, input_size)

            optimizer.zero_grad()  # Zero the gradients
            h = model.init_hidden(batch_size).to(device)
            outputs, _ = model(input1_l, input_s, input_cur, h)  # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate loss

        epoch_duration = time.time() - start_time  # Calculate epoch duration
        avg_epoch_loss = running_loss / dataloader.__len__()
        epoch_losses.append(avg_epoch_loss)

        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Duration: {epoch_duration:.2f} seconds, Learning Rate: {current_lr:.8f}")

        scheduler.step()  # Update learning rate

    print("Training completed.")

    # Save the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = f"{model_save_path}_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

    # Plot the loss
    plot_loss(epoch_losses, num_epochs, result_folder, timestamp)

    return model_save_path


def plot_loss(epoch_losses, num_epochs, result_folder, timestamp):
    """
    Plot and save the training loss.

    Parameters:
        epoch_losses (list): List of loss values for each epoch.
        num_epochs (int): Number of training epochs.
        result_folder (str): Folder to save the plot.
        timestamp (str): Timestamp for saving the plot.
    """
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a JPEG file
    plot_save_path = os.path.join(result_folder, f'training_loss_{timestamp}.jpg')
    plt.savefig(plot_save_path, format='jpg')
    plt.close()
    print(f"Plot saved to {plot_save_path}")


def test(dataloader_test, model, device, model_save_path, result_folder):
    """
    Test the RNN model.

    Parameters:
        dataloader_test (DataLoader): DataLoader for testing data.
        model (nn.Module): The RNN model to test.
        device (torch.device): Device to use for testing (CPU or GPU).
        model_save_path (str): Path to the saved model.
        result_folder (str): Folder to save the result plots.

    Returns:
        tuple: Average test loss, total running loss, outputs, targets.
    """
    print("Start testing------------------------------")

    # Load the model
    try:
        model.load_state_dict(torch.load(model_save_path))
    except FileNotFoundError:
        print(f"Error: Model file {model_save_path} not found.")
        return

    model.to(device)
    model.eval()

    criterion_test = nn.MSELoss()
    all_outputs = []
    all_targets = []
    running_loss_test = 0.0

      # Disable gradient calculation for evaluation
    for inputs, targets in dataloader_test:
        inputs, targets = inputs.to(device), targets.to(device)

        # Split inputs into input1 and input2
        input1_l = inputs[:, :-1, :]  # Shape: (batch_size, sequence_length - 1, input_size)
        input_s = inputs[:, -10:, :]
        input_cur = inputs[:, -1:, :]  # Shape: (batch_size, 1, input_size)
        h = model.init_hidden(batch_size).to(device)
        outputs, _= model(input1_l, input_s, input_cur, h)  # Forward pass

        all_outputs.extend(outputs.cpu().detach().numpy().flatten())
        all_targets.extend(targets.cpu().detach().numpy().flatten())

    rmse = calculate_rmse(all_outputs, all_targets)
    print(f'RMSE: {rmse:.4f}')

    num_samples = len(all_outputs)  # Total samples in the dataset

    # Plotting
    plt.figure()
    plt.plot(range(1, num_samples + 1), all_outputs, label='Test result')
    plt.plot(range(1, num_samples + 1), all_targets, label='True result')
    plt.xlabel('Number')
    plt.ylabel('SOC')
    plt.title(f'True and Test Results {rmse}')
    plt.legend()
    plt.grid(True)

    # Save the plot as a JPEG file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_save_path = os.path.join(result_folder, f'test_result_{timestamp}.png')
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


def plot_results(outputs, targets, num_samples, result_folder, average_loss):
    """
    Plot and save the test results.

    Parameters:
        outputs (np.ndarray): Predicted values.
        targets (np.ndarray): True values.
        num_samples (int): Number of samples.
        result_folder (str): Folder to save the plot.
    """
    plt.figure()
    plt.plot(range(1, num_samples + 1), outputs, label='Test Results', color='r')
    plt.plot(range(1, num_samples + 1), targets, label='True Results', color='b')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Test Results vs. True Results')
    plt.legend()
    plt.grid(True)

    # Save the plot as a JPEG file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_save_path = os.path.join(result_folder, f'test_results_{timestamp}_{average_loss}.jpg')
    plt.savefig(plot_save_path, format='jpg')
    plt.close()
    print(f"Plot saved to {plot_save_path}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Battery soc estimator based on transformer')

    # Parameters
    model_save_path = "model/"
    csv_file_train = 'data/m_train.csv'
    csv_file_test = 'data/m_test.csv'
    result_folder = "results/"
    window_size = 501
    step_size = 1
    batch_size = 1024
    num_epochs = 200
    initial_lr = 0.001
    gamma = 0.9
    step_size_lr = 5

    # define dataset
    dataset_train = SlidingWindowDataset(csv_file_train, window_size, step_size)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last = True)

    dataset_test = SlidingWindowDataset(csv_file_test, window_size, step_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False, pin_memory=True, num_workers=4, drop_last = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The program will run on: {device}')

    model = CustomRNN()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=step_size_lr, gamma=gamma)
    scaler = GradScaler()

    # model_save_path_ = train(model, dataloader_train, criterion, optimizer, scheduler, num_epochs, device,
    #                         model_save_path, result_folder)
    #
    # test(dataloader_test, model, device, model_save_path_, result_folder)
    test(dataloader_test, model, device, 'model/_model_20240808_201222.pth', result_folder)
