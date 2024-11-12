# GeneralisationBounds_SLT

## environment

conda create -n SLT python wandb
conda activate SLT
conda install -c conda-forge lightning pytorch torchvision
pip install --upgrade wandb

## run the code

Currently trying to run it with the following command:
`python3 main.py --dataset_name CIFAR10 --model_name AlexNet --batch_size 32 --lr 0.001 --max_epochs 2>`
