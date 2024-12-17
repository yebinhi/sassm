import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from mamba import Mamba, ModelArgs
from TwinHybridModel import TwinHybridModel
from SlidingWindowDataset import SlidingWindowDataset
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, result_folder):

    epoch_losses = []

    print("------------------------------- Start Training --------------------------------------")
    model.train()

    for epoch in range(num_epochs):
          # Set model to training mode
        running_loss = 0.0
        start_time = time.time()  # Start time for the epoch
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Split inputs into input1 and input2
            input1_l = inputs[:, :-1, :]  # Shape: (batch_size, sequence_length - 1, input_size)
            input_s = inputs[:, -50:, :]
            input_cur = inputs[:, -1:, :]  # Shape: (batch_size, 1, input_size)


            optimizer.zero_grad()  # Zero the gradients
            outputs = model(input1_l, input_s, input_cur)  # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Compute loss
            loss.backward()  # Backward pass
            running_loss += loss.item()  # Accumulate loss
            optimizer.step()  # Update model parameters

        scheduler.step()

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
    model_save_path = f"{model_save_path}_model_sassm_{timestamp}.pth"
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


def calculate_metrics(all_outputs, all_targets):
    """
    Calculate MSE, RMSE, and R^2 for the model predictions.

    Parameters:
        all_outputs (np.ndarray): Model predictions.
        all_targets (np.ndarray): True target values.

    Returns:
        dict: Dictionary containing MSE, RMSE, and R^2.
    """
    mse = mean_squared_error(all_targets, all_outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_outputs)

    return {'MSE': mse, 'RMSE': rmse, 'R^2': r2}

def add_noise(inputs, noise_level=0.01):
    """
    Add Gaussian noise to the inputs.

    Parameters:
        inputs (torch.Tensor): The input tensor to which noise will be added.
        noise_level (float): The standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy inputs.
    """
    noise = torch.randn_like(inputs) * noise_level
    return inputs + noise

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

    for inputs, targets in dataloader_test:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = add_noise(inputs, noise_level=0.0005)

        # Split inputs into input1 and input2
        input1_l = inputs[:, :-1, :]  # Shape: (batch_size, sequence_length - 1, input_size)
        input_s = inputs[:, -50:, :]
        input_cur = inputs[:, -1:, :]  # Shape: (batch_size, 1, input_size)

        outputs = model(input1_l, input_s, input_cur)  # Forward pass
        loss = criterion_test(outputs.squeeze(), targets)
        running_loss_test += loss.item()

        all_outputs.append(outputs.cpu().detach())  # Move to CPU and detach
        all_targets.append(targets.cpu().detach())  # Move to CPU and detach

    num_samples = len(dataloader_test.dataset)  # Total number of samples in the dataset
    average_loss = running_loss_test / num_samples
    print(f"Test Loss: {average_loss:.6f}")

    # Concatenate into single tensors
    all_output = torch.cat(all_outputs, dim=0)
    all_target = torch.cat(all_targets, dim=0)

    # Convert tensors to numpy arrays for plotting
    all_outputs = all_output.numpy()
    all_targets = all_target.numpy()

    # Plotting
    plot_results(all_outputs, all_targets, num_samples, result_folder, average_loss)

    metrics = calculate_metrics(all_outputs, all_targets)
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R^2: {metrics['R^2']:.6f}")

    return average_loss, running_loss_test, all_outputs, all_targets


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
    plot_save_path = os.path.join(result_folder, f'test_results_noise__sassm_mamba_n10_{timestamp}_{average_loss}.jpg')
    combined_array = np.column_stack((outputs, targets))
    np.savetxt(plot_save_path+'.txt', combined_array, delimiter=',', fmt='%.7f')
    plt.savefig(plot_save_path, format='jpg')
    plt.close()
    print(f"Plot saved to {plot_save_path}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Battery soc estimator based on transformer')

    # Parameters
    model_save_path = "model/"
    csv_file_train = 'data/m_train.csv'
    csv_file_test = 'data/test/xy_test_n10.csv'
    result_folder = "results/noise/"
    window_size = 501
    step_size = 1
    batch_size = 8192
    num_epochs = 50
    initial_lr = 0.01
    gamma = 0.95
    step_size_lr = 5

    # define dataset
    dataset_train = SlidingWindowDataset(csv_file_train, window_size, step_size)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    dataset_test = SlidingWindowDataset(csv_file_test, window_size, step_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The program will run on: {device}')

    arg = ModelArgs(d_input=9, d_output=1, d_model=64, n_layer=1)
    model = Mamba(arg)
    model = model.to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=initial_lr, alpha=gamma)
    scheduler = StepLR(optimizer, step_size=step_size_lr, gamma=gamma)

    # model_save_path_ = train(model, dataloader_train, criterion, optimizer, scheduler, num_epochs, device,
    #                          model_save_path, result_folder)
    #
    # test(dataloader_test, model, device, model_save_path_, result_folder)
    test(dataloader_test, model, device, 'model/_model_sassm_20240924_101038.pth', result_folder)
