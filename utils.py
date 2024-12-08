# utils.py
import torch
import math
import numpy as np
import torchvision
import argparse

# def max_pixel_sums(data_loader):
#     """
#     Compute the maximum pixel sum for squared images in the dataset.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
#     # Get image size from the first batch
#     first_batch = next(iter(data_loader))
#     images, _ = first_batch
#     image_size = images[0].size()

#     # Initialize a tensor to hold the pixel sums
#     pixel_sums = torch.zeros(image_size, device=device)

#     # Loop over all images in the dataset and accumulate pixel sums
#     for images, _ in data_loader:
#         images = images.to(device)
#         pixel_sums += torch.sum(torch.pow(images, 2), dim=0)
    
#     #print(f"Pixel sums device: {pixel_sums.device}")
#     #print(f"Images device: {images.device}")

#     # Return the max value of pixel sums
#     return pixel_sums.max().item()

def max_pixel_sums(dataset):
    """
    Compute the maximum pixel sum for squared images in the dataset by iterating over each image.

    Args:
        dataset: PyTorch Dataset object.

    Returns:
        float: Maximum sum of squared pixels across all images in the dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if the dataset is a Subset
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
        dataset = dataset.dataset  # Access the underlying dataset
    else:
        indices = range(len(dataset))  # All indices for non-Subset datasets

    # Get the size of the first image
    image, _ = dataset[indices[0]]
    image_size = image.size()

    # Initialize a tensor to hold the pixel sums
    pixel_sums = torch.zeros(image_size, device=device)

    # Loop through all images in the dataset and compute squared pixel sums
    for i in range(len(dataset)):
        image, _ = dataset[i]  # No need to transform; already a tensor
        pixel_sums += torch.pow(image, 2)

    # Return the maximum value from the summed pixel tensor
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

# def our_total_bound(net, data_loader, num_classes, dataset_size, depth, delta=0.001):
#     """
#     Compute the generalization bound for the given model and return intermediate values.
#     """
#     rho = eval_rho(net)
#     n = dataset_size
#     k = num_classes
#     max_deg = 2  # Assuming ReLU activations (degree 2)
#     deg_prod = max_deg ** depth  # Approximation for product of degrees

#     # Multiplier terms
#     mult1 = (rho + 1) / n
#     mult2 = 2 ** 1.5 * (1 + math.sqrt(2 * (depth * np.log(2 * max_deg) + np.log(k))))
#     max_sum_sqrt = math.sqrt(max_pixel_sums(data_loader))
#     mult3 = max_sum_sqrt * math.sqrt(deg_prod)

#     # Additional term
#     add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))

#     # Final bound
#     bound = mult1 * mult2 * mult3 + add1
#     return bound, mult1, mult2, mult3, add1

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


