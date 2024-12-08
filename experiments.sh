#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main.py"

# Define parameter arrays
train_subset_fractions=(0.25 0.5 0.75 1)
random_label_fractions=(0 0.25 0.5 0.75 1)
noise_image_fractions=(0 0.5 1)
seeds=(12 10 24)

# Automatically detect available GPUs
gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader))
num_gpus=${#gpus[@]}

if [ $num_gpus -eq 0 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

echo "Detected GPUs: ${gpus[@]}"
echo "Number of GPUs: $num_gpus"

# Generate all parameter combinations
combinations=()
for seed in "${seeds[@]}"; do
    for train_subset_fraction in "${train_subset_fractions[@]}"; do
        if (( $(echo "$train_subset_fraction == 1" | bc -l) )); then
            for noise_image_fraction in "${noise_image_fractions[@]}"; do
                combinations+=("$seed None $noise_image_fraction")
            done
        else
            for random_label_fraction in "${random_label_fractions[@]}"; do
                combinations+=("$seed $random_label_fraction None")
            done
        fi
    done
done

# Distribute combinations across GPUs and run tasks
num_combinations=${#combinations[@]}
for ((i = 0; i < num_combinations; i++)); do
    gpu_id=${gpus[$((i % num_gpus))]}  # Cycle through available GPUs
    params=(${combinations[$i]})
    seed=${params[0]}
    random_label_fraction=${params[1]}
    noise_image_fraction=${params[2]}

    # Construct and run the Python command
    echo "Running on GPU $gpu_id: seed=$seed, random_label_fraction=$random_label_fraction, noise_image_fraction=$noise_image_fraction"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python $SCRIPT_PATH \
        --seed $seed \
        --random_label_fraction $random_label_fraction \
        --noise_image_fraction $noise_image_fraction > "gpu_${gpu_id}_run_${i}.log" 2>&1 &
done

# Wait for all processes to finish
wait
echo "All tasks completed."