import os
import torch
import numpy as np
import random
import argparse
import wandb
import sys
import tempfile


import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from data import DataModule
from models import SparseDeepModel, ModularCNN
from utils import none_or_float, simulate_model_dimensions, calculate_total_params, create_kernel_dict, max_pixel_sums, MetricsCallback, CustomEarlyStopping
import plotting

wandb.login(key = '3c5767e934e3aa77255fc6333617b6e0a2aab69f')

def set_seed(seed):
    seed_everything(seed, workers=True)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):

    print('CUDA GPUs present?', torch.cuda.is_available())
    print(torch.cuda.current_device())  # Should print the device index
    print(torch.cuda.get_device_name(0))  # Should print the GPU name
    torch.set_float32_matmul_precision('medium')


    saving_dir = os.getcwd()
    os.environ['TMPDIR'] = os.path.join(os.getcwd(), 'Tempdir')
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    print("Temporary directory set to:", tempfile.gettempdir())
    os.environ['WANDB_DIR'] = os.path.join(os.getcwd(), 'Wandbdir')
    os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
    print("Setting WANDB_DIR to:", os.environ['WANDB_DIR'])

    set_seed(args.seed)

    if args.log_wandb:
        wandb_logger = WandbLogger(project=args.project_name,
                                   entity="SLT_poggio24",
                                   log_model=False, 
                                   save_dir = os.path.join(saving_dir, 'model_logs'))
        wandb_logger.log_hyperparams(args)
    else:
        wandb_logger = None

    filename_parts = [
        f"sd={args.seed}",
        f"dataset={args.dataset_name}",
        f"random_label_fraction={args.random_label_fraction}",
        f"train_subset_fraction={args.train_subset_fraction}",
    ]
    unique_dir_name = "_".join(filename_parts)



    #define dataset
    data_module = DataModule(dataset_name=args.dataset_name, 
                             batch_size=args.batch_size, 
                             random_label_fraction=args.random_label_fraction, 
                             noise_image_fraction = args.noise_image_fraction, 
                             train_subset_fraction = args.train_subset_fraction,
                             val_subset_fraction = args.val_subset_fraction)
    data_module.setup()

    #define model
    kernel_dict = create_kernel_dict(args.kernel_sizes, args.out_channels, args.strides, args.paddings)
    max_pool_layer_dict = {}
    dropout_layer_dict = {}

    if args.model_type == 'ModularCNN':
        model = ModularCNN(kernel_dict=kernel_dict, 
                           max_pool_layer_dict=max_pool_layer_dict, 
                           dropout_layer_dict=dropout_layer_dict, 
                           num_classes= data_module.num_classes, 
                           lr=args.lr, 
                           weight_decay=args.weight_decay, 
                           optimizer_choice=args.optimizer_choice, 
                           momentum=args.momentum, 
                           use_warmup=args.use_warmup, 
                           lr_decay_type=args.lr_decay_type, 
                           warmup_steps = args.max_epochs/5, 
                           max_steps = args.max_epochs,
                           weight_init=args.weight_init)
        
        
        
    elif args.model_type == 'LegacyModels':
        model = SparseDeepModel(model_name=args.model_name, 
                                num_classes= data_module.num_classes, 
                            lr=args.lr, 
                            weight_decay=args.weight_decay)

    

    # Compute max pixel sums for bound computation
    max_pixel_sum = max_pixel_sums(args.dataset_name)
    print(f"Max Pixel Sums : {max_pixel_sum}")
    model.max_pixel_sum = max_pixel_sum

    #define callbacks
    callbacks = []

    if args.model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',        # Ensure this is the exact name used in your logging
            dirpath= os.path.join(saving_dir, 'model_checkpoints', unique_dir_name),  # Directory to save checkpoints
            filename=f'best-{{epoch:02d}}-{{val_loss:.2f}}-{unique_dir_name}',
            save_top_k=1,
            mode='min',                     # Minimize the monitored value
            save_last=True,                # Save the last model to resume training
            verbose = True
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping and args.random_label_fraction is None:
        early_stopping = EarlyStopping(
            min_delta=0.00,
            monitor='train_acc',
            patience=150,
            mode='max',
            verbose=True
        )
        callbacks.append(early_stopping)
        
        early_stopping_callback = CustomEarlyStopping(target_accuracy=0.99)
        callbacks.append(early_stopping_callback)

    metrics_callback = MetricsCallback(unique_dir_name)
    callbacks.append(metrics_callback)    


    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        #fast_dev_run = True,
        #overfit_batches = 1
        #deterministic=True,
        #check_val_every_n_epoch=1,
        devices=1,
        #strategy=“ddp”,
        #accumulate_grad_batches=1,
        #profiler='simple'   #this helps to identify bottlenecks
    )
    trainer.fit(model, data_module)

    trainer.test(model, ckpt_path='last', dataloaders = data_module.test_dataloader())

    #plotting.plot_bound_vs_epochs(unique_dir_name)
    #plotting.plot_bound_vs_train_subset_fraction(unique_dir_name)
    #plotting.plot_layer_norms_vs_epochs(unique_dir_name)
    #plotting.plot_bound_vs_hyperparams(unique_dir_name)
    #plotting.plot_bound_vs_optimizer(unique_dir_name)

if __name__ == '__main__':
    #sys.stdout = open('SLT_project_output.txt', 'w')

    parser = argparse.ArgumentParser(description="Train a model on CV dataset")

    parser.add_argument('--project_name', type=str, default='SLT_experiments_1', help='Name of the wandb project')
    parser.add_argument('--log_wandb', action='store_true', help='Enable logging to wandb')
    parser.add_argument('--no_log_wandb', dest='log_wandb', action='store_false', help='Disable logging to wandb')
    parser.set_defaults(log_wandb=True)

    # Experiment specific args 
    parser.add_argument('--dataset_name', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'ImageNet'], help='Dataset to use')
    parser.add_argument('--train_subset_fraction', type=float, default=1.0, help='Size of the training subset to use')
    parser.add_argument('--val_subset_fraction', type=float, default=1.0, help='Size of the validation subset to use')
    parser.add_argument('--random_label_fraction', type=none_or_float, default=None, help='Fraction of labels to randomize in the training dataset. Must be between 0.0 and 1.0, or None.')
    parser.add_argument('--noise_image_fraction', type=none_or_float, default=None, help='Fraction of noise to add to training data. Must be between 0.0 and 1.0, or None.')
    
    # Model-specific args
    parser.add_argument('--model_type', type=str, default='ModularCNN', choices=['ModularCNN', 'LegacyModels'], help='Model type to use')
    parser.add_argument('--model_name', type=str, default='AlexNet', choices=['AlexNet', 'InceptionV3'], help='Legacy model to use')
    
    # Architecture arguments
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[2, 2, 2], help='List of kernel sizes for each layer')
    parser.add_argument('--out_channels', nargs='+', type=int, default=[200,200, 200], help='List of output channels for each layer')
    parser.add_argument('--strides', nargs='+', type=int, default=[1, 1, 1], help='List of strides for each layer')
    parser.add_argument('--paddings', nargs='+', type=int, default=[0, 0, 0], help='List of paddings for each layer')
   
    parser.add_argument('--weight_init', action='store_true', help='Enable weight initialization')
    parser.add_argument('--no_weight_init', dest='weight_init', action='store_false', help='Disable weight initialization')
    parser.set_defaults(weight_init=False)
    
    # Trainer-specific args
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs to train')
    
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer_choice', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=8e-4, help='Weight decay')

    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--use_warmup', action='store_true', help='Enable lr warmup')
    parser.add_argument('--no_use_warmup', dest='use_warmup', action='store_false', help='Disable lr warmup')
    parser.set_defaults(use_warmup=False)
    parser.add_argument('--lr_decay_type', type=str, default='multi_step', choices=['multi_step', 'cosine', 'linear'], help='Type of lr decay to use')    
    
    parser.add_argument('--model_checkpoint', action='store_true', help='Enable model checkpointing')
    parser.add_argument('--no_model_checkpoint', dest='model_checkpoint', action='store_false', help='Disable model checkpointing')
    parser.set_defaults(model_checkpoint=False)
    
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false', help='Disable early stopping')
    parser.set_defaults(early_stopping=True)

    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which accelerator to use')

    
     
    args = parser.parse_args()


    if args.dataset_name == 'MNIST':
        input_dims = (28, 28, 1)
        num_classes = 10
    elif args.dataset_name == 'CIFAR10':
        input_dims = (28, 28, 3)
        num_classes = 10
    elif args.dataset_name == 'ImageNet':
        input_dims = (224, 224, 3)
        num_classes = 1000

    calculate_total_params(args, input_dims=input_dims, num_classes=num_classes)
    simulate_model_dimensions(args.kernel_sizes, args.out_channels, args.strides, args.paddings, input_dims, args.dataset_name)
    
    
    main(args)
