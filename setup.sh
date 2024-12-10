#!/bin/bash

# Navigate to the parent directory
cd ..

# Download and install Miniconda
echo "Downloading and installing Miniconda..."
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Initialize Miniconda
echo "Initializing Miniconda..."
export PATH=$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/bin/activate

# Configure Conda
echo "Configuring Conda..."
conda config --set auto_activate_base false

# Navigate to the project folder
cd GeneralisationBounds_SLT
echo "Switching to the GPU-testing branch..."
git switch GPU-testing

# Set up the Conda environment
echo "Creating and activating the Conda environment..."
conda env create -f env.yaml
conda activate SLT

echo "Environment setup complete. You are now in the GPU-testing branch and the SLT environment is activated."