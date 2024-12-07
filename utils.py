# utils.py
import torch
import math
import numpy as np
import torchvision

def max_pixel_sums(data_loader):
    """
    Compute the maximum pixel sum for squared images in the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Get image size from the first batch
    first_batch = next(iter(data_loader))
    images, _ = first_batch
    image_size = images[0].size()

    # Initialize a tensor to hold the pixel sums
    pixel_sums = torch.zeros(image_size, device=device)

    # Loop over all images in the dataset and accumulate pixel sums
    for images, _ in data_loader:
        images = images.to(device)
        pixel_sums += torch.sum(torch.pow(images, 2), dim=0)
    
    #print(f"Pixel sums device: {pixel_sums.device}")
    #print(f"Images device: {images.device}")

    # Return the max value of pixel sums
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

def our_total_bound(net, data_loader, num_classes, dataset_size, depth, delta=0.001):
    """
    Compute the generalization bound for the given model and return intermediate values.
    """
    rho = eval_rho(net)
    n = dataset_size
    k = num_classes
    max_deg = 2  # Assuming ReLU activations (degree 2)
    deg_prod = max_deg ** depth  # Approximation for product of degrees

    # Multiplier terms
    mult1 = (rho + 1) / n
    mult2 = 2 ** 1.5 * (1 + math.sqrt(2 * (depth * np.log(2 * max_deg) + np.log(k))))
    max_sum_sqrt = math.sqrt(max_pixel_sums(data_loader))
    mult3 = max_sum_sqrt * math.sqrt(deg_prod)

    # Additional term
    add1 = 3 * math.sqrt(np.log((2 * (rho + 2) ** 2) / delta) / (2 * n))

    # Final bound
    bound = mult1 * mult2 * mult3 + add1
    return bound, mult1, mult2, mult3, add1