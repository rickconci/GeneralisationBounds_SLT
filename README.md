# GeneralisationBounds_SLT

### Abstract

Recent advancements in statistical learning theory have introduced tighter general-
ization bounds for deep neural networks with compositional sparsity, challenging
the long-held critique of traditional complexity measures. In this work, we re-
visit the relationship between theoretical bounds and empirical performance in
the context of convolutional neural networks (CNNs). We empirically evaluate
tighter generalization bounds focusing on networks trained with varying degree
of randomness showing that, while still vacous, they entail great information
about the generalization performance of the network. We also investigate how the
bound changes with scaled data sizes, indicating with extended data, a non-vacous
bound could be achieved. Lastly, we offer insight to how regularization strategies
and hyperparameters can be used in future work to generate tight bounds. Our
findings underscore the potential of architecture-specific norm-based bounds in
generalization capabilities of CNNs

### Setup

**Download and install Miniconda**
`curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
`sh Miniconda3-latest-Linux-x86_64.sh -b`  # -b flag to install without prompts
`source ~/.bashrc`
`conda config --set auto_activate_base false`

**Environment setup**
`conda env create -f env2.yaml`
`conda activate SLT`

**env composed of:**
`conda create -n SLT python=3.10`
`conda activate SLT`
`conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
`python -m pip install lightning`
`conda install conda-forge::wandb`
`pip install gpustat`

**for GPU monitoring:**
`gpustat --watch`

**for GPU killing:**
`nvidia-smi | grep 'python' | awk '{print $5}' | xargs -r kill -9`

**Run it with the following command:**
`python3 main.py`

**Parallel GPU Experiments**
`sh_folder/arch_search.sh`
`sh_folder/SLT_experiment3.sh`

Plotting code not included in the repository.
