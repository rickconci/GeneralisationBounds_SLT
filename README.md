# GeneralisationBounds_SLT

## environment

cd into home folder
`curl -O <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh>`
`sh Miniconda3-latest-Linux-x86_64.sh`
`source ~/.bashrc`
`conda config --set auto_activate_base false`

cd into the project folder
`conda env create -f env.yaml`

switch to correct branch
`git checkout <branch_name>`

env composed of:
`conda create -n SLT python=3.10`
`conda activate SLT`
`conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
`python -m pip install lightning`
`conda install conda-forge::wandb`
`pip install gpustat`

for GPU monitoring:
`gpustat --watch`

## run the code

Currently trying to run it with the following command:
`python3 main.py`
