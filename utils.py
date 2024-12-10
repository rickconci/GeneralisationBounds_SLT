# utils.py
import torch
import math
import numpy as np
import torchvision
import argparse
import os
import pandas as pd
from lightning.pytorch.callbacks import Callback

class MetricsCallback(Callback):
    def __init__(self, experiment_name, save_dir='experiment_data'):
        super().__init__()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.metrics_data = {
            'bound_data': [],
            'layer_norms': [],
            'hyperparameter_bounds': []  # New list for hyperparameter-specific bound data
        }
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect data every epoch
        epoch = trainer.current_epoch
        
        # Get experiment parameters
        random_label_fraction = trainer.datamodule.random_label_fraction
        train_subset_fraction = trainer.datamodule.train_subset_fraction
        
        # Get additional hyperparameters
        weight_decay = pl_module.hparams.weight_decay
        batch_size = trainer.datamodule.batch_size
        optimizer_type = pl_module.hparams.optimizer_choice
        
        # Get bound data
        bound = trainer.callback_metrics.get('generalization_bound', None)
        if bound is not None:
            # Existing bound data collection
            self.metrics_data['bound_data'].append({
                'epoch': epoch,
                'random_label_fraction': random_label_fraction,
                'train_subset_fraction': train_subset_fraction,
                'bound': bound.item()
            })
            
            # New hyperparameter-specific bound data collection
            self.metrics_data['hyperparameter_bounds'].append({
                'epoch': epoch,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'optimizer_type': optimizer_type,
                'bound': bound.item()
            })

        # Get layer norms
        for idx, layer in enumerate(pl_module.weight_layers):
            norm = torch.norm(layer.weight, p='fro').item()
            self.metrics_data['layer_norms'].append({
                'epoch': epoch,
                'layer_idx': idx,
                'random_label_fraction': random_label_fraction,
                'train_subset_fraction': train_subset_fraction,
                'norm': norm
            })

    def on_train_end(self, trainer, pl_module):
        # Save collected data to pickle files
        bound_df = pd.DataFrame(self.metrics_data['bound_data'])
        norms_df = pd.DataFrame(self.metrics_data['layer_norms'])
        hyperparameter_bounds_df = pd.DataFrame(self.metrics_data['hyperparameter_bounds'])
        
        # Save DataFrames
        bound_df.to_pickle(os.path.join(self.save_dir, f'{self.experiment_name}_bounds.pkl'))
        norms_df.to_pickle(os.path.join(self.save_dir, f'{self.experiment_name}_norms.pkl'))
        hyperparameter_bounds_df.to_pickle(os.path.join(self.save_dir, f'{self.experiment_name}_hyperparameter_bounds.pkl'))


class CustomEarlyStopping(Callback):
    def __init__(self, target_accuracy=0.95, max_epochs=3000):
        '''
        Custom early stopping callback.
        Args:
            target_accuracy (float): Training will stop when this accuracy is reached.
            max_epochs (int): Maximum number of epochs to train if target accuracy is not reached.
        '''
        self.target_accuracy = target_accuracy
    def on_train_epoch_end(self, trainer, pl_module):
        '''
        Called at the end of each training epoch.
        Args:
            trainer: The Lightning Trainer.
            pl_module: The LightningModule (your model).
        '''
        # Get the logged training accuracy
        train_acc = trainer.callback_metrics.get('train_acc', None)
        # Stop training if target accuracy is reached
        if train_acc is not None and train_acc >= self.target_accuracy:
            print(f'Training stopped early as training accuracy reached {train_acc:.4f}.')
            trainer.should_stop = True



def max_pixel_sums(dataset_name):
    """
    Compute the maximum pixel sum for squared images in the dataset by iterating over each image.

    Args:
        dataset: PyTorch Dataset object.

    Returns:
        float: Maximum sum of squared pixels across all images in the dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    image, _ = dataset[0]
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_size = image_tensor.size()

    # Initialize a tensor to hold the pixel sums
    pixel_sums = torch.zeros(image_size, device=device)

    # Loop over all images in the dataset and add their pixels to the sums
    for i in range(len(dataset)):
        image, _ = dataset[i]
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        pixel_sums += torch.pow(image_tensor, 2)

    # Return the tensor of pixel sums
    return pixel_sums.max().item()


def eval_rho(net):
    """
    Compute the product of Frobenius norms of all layers in the network.
    """
    rho = 1
    for layer in net.modules():  # Iterates through all layers
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            rho *= torch.norm(layer.weight.data, p='fro').item()
    return rho


def our_total_bound(net, num_classes, dataset_size, depth, kernel_dict, max_pixel_sum, delta=0.001):
    """
    Compute the generalization bound for the given model and return intermediate values.

    Args:
        net: The neural network.
        data_loader: Data loader for the dataset.
        num_classes: Number of classes in the dataset.
        dataset_size: Number of samples in the dataset.
        depth: Depth of the network (number of layers with parameters).
        kernel_dict: Dictionary of convolutional layer configurations.
        delta: Small constant for numerical stability.

    Returns:
        bound, mult1, mult2, mult3, add1: Generalization bound and intermediate values.
    """
    # Compute rho (product of Frobenius norms of all layers)
    rho = eval_rho(net)
    n = dataset_size
    k = num_classes

    kernel_sizes = np.array([params['kernel_size'] for params in kernel_dict.values()])
    #max_deg = max(kernel_sizes)**2 
    deg_prod = np.prod(kernel_sizes)

    mult1 = (rho + 1) / n
    mult2 = 2 ** 1.5 * (1 + math.sqrt(2 * (depth * np.log(2) + sum(np.log((kernel_sizes)**2)) + np.log(k))))
    mult3 = math.sqrt(max_pixel_sum) * deg_prod
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))

    # Final bound
    bound = mult1 * mult2 * mult3 + add1
    return bound, mult1, mult2, mult3, add1



def simulate_model_dimensions(kernel_sizes, out_channels, strides, paddings, input_dims, dataset_name="MNIST"):
    """
    Simulates the dimensions of the data as it passes through the model layers.
    
    Args:
        kernel_sizes (list): List of kernel sizes for the convolutional layers.
        out_channels (list): List of output channels for the convolutional layers.
        strides (list): List of strides for the convolutional layers.
        paddings (list): List of paddings for the convolutional layers.
        input_dims (tuple): A tuple (height, width, channels) representing the input dimensions.
        dataset_name (str): Name of the dataset (e.g., "MNIST", "CIFAR10").
    """
    h, w, in_channels = input_dims
   
    print(f"Dataset: {dataset_name}")
    print(f"Input Dimensions: Height={h}, Width={w}, Channels={in_channels}\n")
    
    num_layers = len(kernel_sizes)
    
    for layer_idx in range(num_layers):
        print(f"Layer {layer_idx}:")
        
        # Convolutional Layer
        if layer_idx < len(kernel_sizes):
            kernel_size = kernel_sizes[layer_idx]
            out_channel = out_channels[layer_idx]
            stride = strides[layer_idx]
            padding = paddings[layer_idx]
            
            # Compute output dimensions after convolution
            h_out = (h + 2 * padding - kernel_size) // stride + 1
            w_out = (w + 2 * padding - kernel_size) // stride + 1
            
            print(f"  Conv2d: kernel_size={kernel_size}, stride={stride}, padding={padding}")
            print(f"    Input Channels: {in_channels}")
            print(f"    Output Channels: {out_channel}")
            print(f"    Output Dimensions: Height={h_out}, Width={w_out}")
            
            # Update dimensions and channels for next layer
            h, w = h_out, w_out
            in_channels = out_channel
            
        print("")  # Empty line for readability
    
    # After all layers
    num_flat_features = in_channels * h * w
    print(f"Final Output Dimensions before Linear Layer: {num_flat_features}")
    
    if num_flat_features <= 0:
        print("Error: The number of features before the linear layer is non-positive.")
        print("Possible causes:")
        print("  - The dimensions have been reduced too much due to convolution or pooling.")
        print("  - Incorrect padding, stride, or kernel sizes leading to zero or negative dimensions.")
    else:
        print("The dimensions are valid for the linear layer input.")

    return num_flat_features



def calculate_total_params(args, input_dims, num_classes=10):
    total_params = 0
    current_height, current_width, in_channels = input_dims

    # Calculate parameters for convolutional layers
    for i in range(len(args.kernel_sizes)):
        kernel_size = args.kernel_sizes[i]
        out_channels = args.out_channels[i]
        stride = args.strides[i]
        padding = args.paddings[i]

        # Parameters: (kernel_size * kernel_size * in_channels + 1) * out_channels
        conv_params = (kernel_size * kernel_size * in_channels + 1) * out_channels
        total_params += conv_params

        # Update feature map size
        current_height = math.floor((current_height + 2 * padding - kernel_size) / stride) + 1
        current_width = math.floor((current_width + 2 * padding - kernel_size) / stride) + 1

        # Update in_channels for next layer
        in_channels = out_channels

    # Parameters for the fully connected layer
    fc_input_size = current_height * current_width * in_channels
    fc_params = (fc_input_size + 1) * num_classes  # +1 for bias
    total_params += fc_params

    return total_params



def create_kernel_dict(kernel_sizes, out_channels, strides, paddings):
    """
    Converts lists of kernel parameters into a kernel_dict structure.

    Args:
        kernel_sizes (list): List of kernel sizes for each layer.
        out_channels (list): List of output channels for each layer.
        strides (list): List of strides for each layer.
        paddings (list): List of paddings for each layer.

    Returns:
        dict: A dictionary with layer indices as keys and kernel parameters as values.
    """
    kernel_dict = {}
    num_layers = len(kernel_sizes)
    
    for i in range(num_layers):
        kernel_dict[i] = {
            'kernel_size': kernel_sizes[i],
            'out_channels': out_channels[i],
            'stride': strides[i],
            'padding': paddings[i],
        }
    
    return kernel_dict


def none_or_float(value):
    if value == "None":
        return None
    try:
        float_value = float(value)
        if 0.0 <= float_value <= 1.0:
            return float_value
        else:
            raise argparse.ArgumentTypeError(f"Value must be between 0.0 and 1.0, got {value}.")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be 'None' or a float.")


