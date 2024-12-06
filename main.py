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
from models import SparseDeepModel, ModularCNN, simulate_model_dimensions

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


def main(args, kernel_dict, max_pool_layer_dict, dropout_layer_dict):

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
        f"model={args.model_name}",
        f"random_label_fraction={args.random_label_fraction}",
        f"noise_image_fraction={args.noise_image_fraction}"
    ]
    unique_dir_name = "_".join(filename_parts)



    #define dataset
    data_module = DataModule(dataset_name=args.dataset_name, 
                             batch_size=args.batch_size, 
                             random_label_fraction=args.random_label_fraction, 
                             noise_image_fraction = args.noise_image_fraction, 
                             train_subset_fraction = args.train_subset_fraction,
                             val_subset_fraction = args.val_subset_fraction)

    #define model
    if args.model_type == 'ModularCNN':
        model = ModularCNN(kernel_dict=kernel_dict, 
                           max_pool_layer_dict=max_pool_layer_dict, 
                           dropout_layer_dict=dropout_layer_dict, 
                           num_classes= data_module.num_classes, 
                           lr=args.lr, 
                           weight_decay=args.weight_decay, 
                           warmup_steps = args.max_epochs/5, 
                           max_steps = args.max_epochs)
        
    elif args.model_type == 'LegacyModels':
        model = SparseDeepModel(model_name=args.model_name, 
                                num_classes= data_module.num_classes, 
                            lr=args.lr, 
                            weight_decay=args.weight_decay)

    

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

    if args.early_stopping:
        early_stopping = EarlyStopping(
            min_delta=0.00,
            monitor='val_loss',        # Ensure this is the exact name used in your logging
            patience=100,                    # num epochs with a val loss not improving before it stops 
            mode='min',                     # Minimize the monitored value
            verbose=True
        )
        callbacks.append(early_stopping)

    #callbacks.append[DeviceStatsMonitor()]


    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
        log_every_n_steps=20,
        callbacks=callbacks,
        #fast_dev_run = True,
        #overfit_batches = 1
        #deterministic=True,
        #check_val_every_n_epoch=1,
        devices=1, 
        #strategy="ddp",
        accumulate_grad_batches=6,
        profiler="simple"   #this helps to identify bottlenecks 
    )
    trainer.fit(model, data_module)

    trainer.test(model, ckpt_path='last', dataloaders = data_module.test_dataloader())



if __name__ == '__main__':
    #sys.stdout = open('SLT_project_output.txt', 'w')

    parser = argparse.ArgumentParser(description="Train a model on CV dataset")
    # Experiment specific args 
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', choices=['CIFAR10', 'ImageNet'], help='Dataset to use')
    parser.add_argument('--train_subset_fraction', type=float, default=1.0, help='Size of the training subset to use')
    parser.add_argument('--val_subset_fraction', type=float, default=1.0, help='Size of the validation subset to use')
    
    parser.add_argument('--random_label_fraction', type=float, default=None, help='Fraction of labels to randomize in the training dataset. Must be between 0.0 (no random labels) and 1.0 (all labels randomized.)')
    parser.add_argument('--noise_image_fraction', type=float, default=None, help='Fraction of noise to add to training data. Must be between 0.0 (no noise) and 1.0 (pure noise).')

    # Model specific args
    parser.add_argument('--model_type', type=str, default='ModularCNN', choices=['ModularCNN', 'LegacyModels'], help='Model type to use')
    parser.add_argument('--model_name', type=str, default='AlexNet', choices=['AlexNet', 'InceptionV3'], help='Legacy model to use')

    # Trainer specific args
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum number of epochs to train')
    
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which accelerator to use')

    parser.add_argument('--log_wandb', type=bool, default=True, help='Whether to log to wandb')
    parser.add_argument('--project_name', type=str, default='SLT_project', help='Name of the wandb project')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    parser.add_argument('--model_checkpoint', type=bool, default=False, help='Enable model checkpointing')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Enable early stopping')

    args = parser.parse_args()


    #kernel_dict = {
    #    0: {'kernel_size': 3, 'out_channels': 16, 'stride': 1, 'padding': 1},  # Initial layer
    #    1: {'kernel_size': 3, 'out_channels': 32, 'stride': 2, 'padding': 1},  # Down-sampling via stride
    #    2: {'kernel_size': 3, 'out_channels': 64, 'stride': 2, 'padding': 1},  # Further down-sampling
    #    3: {'kernel_size': 3, 'out_channels': 128, 'stride': 1, 'padding': 1},  # Feature extraction
    #    4: {'kernel_size': 3, 'out_channels': 256, 'stride': 1, 'padding': 1},  # Deeper layer
    #}

    kernel_dict = {
        0: {'kernel_size': 3, 'out_channels': 32, 'stride': 1, 'padding': 1},  # Increased channels
        1: {'kernel_size': 3, 'out_channels': 64, 'stride': 2, 'padding': 1},  # Increased channels
        2: {'kernel_size': 3, 'out_channels': 128, 'stride': 2, 'padding': 1},  # Increased channels
        3: {'kernel_size': 3, 'out_channels': 256, 'stride': 1, 'padding': 1},  # Deeper layer
        4: {'kernel_size': 3, 'out_channels': 512, 'stride': 1, 'padding': 1},  # Increased depth
    }

    max_pool_layer_dict = {}
    #max_pool_layer_dict = {
    #    1: {'pool_size': 2, 'stride': 2}  # Moved to layer 2 and stride set to 2
    #}

    dropout_layer_dict = {
        1: 0.3,  # After the first down-sampling layer (layer 1)
        3: 0.5,  # After layer 3 (feature extraction)
        4: 0.5,  # After layer 4 (deep feature extraction)
    } 

    simulate_model_dimensions(kernel_dict, max_pool_layer_dict, dropout_layer_dict, input_dims=(28, 28, 3))
    
    
    main(args, kernel_dict, max_pool_layer_dict, dropout_layer_dict)
