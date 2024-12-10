import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data

def plot_bound_vs_epochs(experiment_name, save_dir='experiment_data'):
    bounds_df = pd.read_pickle(os.path.join(save_dir, f'{experiment_name}_bounds.pkl'))

    # Plot 1: Generalization bound vs epochs for different random_label_fractions
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=bounds_df, x='epoch', y='bound', hue='random_label_fraction')
    plt.title('Generalization Bound vs Epochs')
    plt.show()

def plot_bound_vs_train_subset_fraction(experiment_name, save_dir='experiment_data'):
    bounds_df = pd.read_pickle(os.path.join(save_dir, f'{experiment_name}_bounds.pkl'))

    # Plot 2: Generalization bound vs train_subset_fraction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=bounds_df.groupby(['train_subset_fraction', 'random_label_fraction'])['bound'].mean().reset_index(),
                x='train_subset_fraction', y='bound', hue='random_label_fraction')
    plt.title('Generalization Bound vs Training Subset Fraction')
    plt.show()

def plot_layer_norms_vs_epochs(experiment_name, save_dir='experiment_data'):
    # Plot 3: Layer norms vs epochs for different random_label_fractions
    norms_df = pd.read_pickle(os.path.join(save_dir, f'{experiment_name}_norms.pkl'))
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=norms_df, x='epoch', y='norm', hue='layer_idx', style='random_label_fraction')
    plt.title('Layer Norms vs Epochs')
    plt.show()


    

def plot_bound_vs_hyperparams(experiment_name, save_dir='experiment_data'):
    """Plot generalization bound vs weight decay and batch size."""
    # Load the data
    df = pd.read_pickle(os.path.join(save_dir, f'{experiment_name}_hyperparameter_bounds.pkl'))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot bound vs weight decay
    sns.lineplot(data=df, x='weight_decay', y='bound', ax=ax1)
    ax1.set_xlabel('Weight Decay')
    ax1.set_ylabel('Generalization Bound')
    ax1.set_title('Bound vs Weight Decay')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot bound vs batch size
    sns.lineplot(data=df, x='batch_size', y='bound', ax=ax2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Generalization Bound')
    ax2.set_title('Bound vs Batch Size')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{experiment_name}_bound_vs_hyperparams.png'))
    plt.close()

def plot_bound_vs_optimizer(experiment_name, save_dir='experiment_data'):
    """Plot generalization bound vs optimizer type."""
    # Load the data
    df = pd.read_pickle(os.path.join(save_dir, f'{experiment_name}_hyperparameter_bounds.pkl'))
    
    plt.figure(figsize=(10, 6))
    
    # Create box plot for bound vs optimizer type
    sns.boxplot(data=df, x='optimizer_type', y='bound')
    plt.xlabel('Optimizer Type')
    plt.ylabel('Generalization Bound')
    plt.title('Bound Distribution by Optimizer Type')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{experiment_name}_bound_vs_optimizer.png'))
    plt.close()