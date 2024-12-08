#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main.py"

# Define architectures
architectures=(
    "3 2 2|1 1 2|0 0 1|128 256 512"   # Gradual growth with efficient strides
    "3 2 2|1 2 2|0 0 0|128 256 512"  # Larger strides in deeper layers for efficiency
    "2 3 3|1 1 2|0 0 1|128 256 512"   # Small kernel size early, larger deeper
    "3 3 3|1 2 2|0 1 1|128 256 512"  # Balanced padding and moderate channel growth
    "3 2 2 2 |1 1 2 1|0 0 1 0|128 128 256 512"  # Additional depth for increased capacity
    "3 2 2|2 2 2|0 0 0|128 256 512"  # Efficient architecture with aggressive downsampling
)

# Define parameter arrays
lr=(0.001)
batch_size=(1248 2048)
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
            for architecture in "${architectures[@]}"; do
                combinations+=("$random_label_fraction $lr $batch_size $architecture")
            done
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
    architecture=${hyperparams[@]:3}  # Extract architecture

    # Determine early_stopping and max_epochs based on random_label_fraction
    if [ "$random_label_fraction" == "None" ]; then
        early_stopping=true
        max_epochs=250
    else
        early_stopping=false
        max_epochs=800
    fi

    # Parse architecture into separate arguments
    IFS='|' read -r kernel_sizes strides paddings out_channels <<< "$architecture"

    # Run the Python script on the assigned GPU
    echo "Starting on GPU $gpu_id with lr=$lr, batch_size=$batch_size, random_label_fraction=$random_label_fraction, early_stopping=$early_stopping, max_epochs=$max_epochs, architecture=($kernel_sizes $strides $paddings $out_channels)"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python $SCRIPT_PATH \
        --lr $lr \
        --batch_size $batch_size \
        --random_label_fraction $random_label_fraction \
        --early_stopping $early_stopping \
        --max_epochs $max_epochs \
        --kernel_sizes $kernel_sizes \
        --strides $strides \
        --paddings $paddings \
        --out_channels $out_channels > "gpu_${gpu_id}_run_${i}.log" 2>&1 &
done

# Wait for all processes to finish
wait
echo "All tasks completed."