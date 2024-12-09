# GeneralisationBounds_SLT

## environment

`git clone https://github.com/rickconci/GeneralisationBounds_SLT.git`
`cd GeneralisationBounds_SLT`
`. setup.sh`

env composed of:
`conda create -n SLT python=3.10`
`conda activate SLT`
`conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
`python -m pip install lightning`
`conda install conda-forge::wandb`
`pip install gpustat`

for GPU monitoring:
`gpustat --watch`

for GPU killing:
`nvidia-smi | grep 'python' | awk '{print $5}' | xargs -r kill -9`

## run the code

Currently trying to run it with the following command:
`python3 main.py`

--

Questions:

- code says multi_step gamma is 0.2, but in the readme it is 0.1.
