#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main.py"

# Define parameter arrays
lr=(0.01 0.005 0.001)
batch_size=(526 1248 2048 4096)
random_label_fraction=("None" 1)

# Automatically detect available GPUs
gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader))
num_gpus=${#gpus[@]}

if [ $num_gpus -eq 0 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

echo "Detected GPUs: ${gpus[@]}"
echo "Number of GPUs: $num_gpus"

# Generate all combinations of hyperparameters
combinations=()
for random_label_fraction in "${random_label_fraction[@]}"; do
    for lr in "${lr[@]}"; do
        for batch_size in "${batch_size[@]}"; do
            combinations+=("$random_label_fraction $lr $batch_size")
        done
    done
done

# Distribute combinations across GPUs
num_combinations=${#combinations[@]}
for ((i = 0; i < num_combinations; i++)); do
    gpu_id=${gpus[$((i % num_gpus))]}  # Cycle through available GPUs
    hyperparams=(${combinations[$i]})  # Extract combination
    random_label_fraction=${hyperparams[0]}
    lr=${hyperparams[1]}
    batch_size=${hyperparams[2]}

    # Run the Python script on the assigned GPU
    echo "Starting on GPU $gpu_id with lr=$lr, batch_size=$batch_size, random_label_fraction=$random_label_fraction"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python $SCRIPT_PATH \
        --lr $lr \
        --batch_size $batch_size \
        --random_label_fraction $random_label_fraction > "gpu_${gpu_id}_run_${i}.log" 2>&1 &
done

# Wait for all processes to finish
wait
echo "All tasks completed."