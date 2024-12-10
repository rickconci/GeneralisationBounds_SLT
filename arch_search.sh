#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main.py"

# Define architectures
# architectures=(
#     "3 2 2|1 1 2|0 0 1|128 256 512"
#     "3 2 2|1 2 2|0 0 0|128 256 512"
#     "2 3 3|1 1 2|0 0 1|128 256 512"
#     "3 3 3|1 2 2|0 1 1|128 256 512"
#     "3 2 2 2|1 1 2 1|0 0 1 0|128 128 256 512"
#     "3 2 2|2 2 2|0 0 0|128 256 512"
# )

architectures=(
    "2 2 2|2 2 2|0 0 0|128 256 512"
)

# Define parameter arrays
lr=(0.010.001)
batch_size=(128 512 1248)
random_label_fraction=("None" 1)
optimizer_choice=('SGD' 'AdamW')

# Automatically detect available GPUs
gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader))
num_gpus=${#gpus[@]}

if [ $num_gpus -eq 0 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

echo "Detected GPUs: ${gpus[@]}"
echo "Number of GPUs: $num_gpus"

# Create a logs directory if it doesn't exist
mkdir -p logs

# Generate all combinations of hyperparameters
combinations=()
for random_label_fraction in "${random_label_fraction[@]}"; do
    for lr_value in "${lr[@]}"; do
        for batch_size_value in "${batch_size[@]}"; do
            for architecture in "${architectures[@]}"; do
                for optimizer_choice in "${optimizer_choice[@]}"; do
                    # Extract architecture components
                    IFS='|' read -r kernel_sizes strides paddings out_channels <<< "$architecture"
                    # Store combination as a single string with a unique delimiter
                    combinations+=("$random_label_fraction|$lr_value|$batch_size_value|$kernel_sizes|$strides|$paddings|$out_channels|$optimizer_choice")
                done
            done
        done
    done
done

num_combinations=${#combinations[@]}
echo "Number of combinations: $num_combinations"

# Define the number of slots per GPU
slots_per_gpu=2  # Adjust this value as needed

# Initialize an associative array to keep track of PIDs per GPU
declare -A gpu_pids

for gpu_id in "${gpus[@]}"; do
    # Get the PIDs of processes running on this GPU
    pids=($(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits --id=$gpu_id))
    gpu_pids[$gpu_id]="${pids[@]}"
done

for ((i = 0; i < num_combinations; i++)); do
    # Assign gpu_id in a round-robin fashion
    gpu_id_index=$((i % num_gpus))
    gpu_id=${gpus[$gpu_id_index]}

    # Wait until the number of jobs on this GPU is less than slots_per_gpu
    while true; do
        pids_str="${gpu_pids[$gpu_id]}"
        pids=($pids_str)
        # Remove finished PIDs
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        # Update the PIDs for this GPU
        gpu_pids[$gpu_id]="${new_pids[@]}"

        if [ "${#new_pids[@]}" -lt "$slots_per_gpu" ]; then
            break
        fi
        sleep 1  # Wait for a second before checking again
    done


    # Get hyperparameters
    IFS='|' read -r random_label_fraction lr batch_size kernel_sizes strides paddings out_channels optimizer_choice <<< "${combinations[$i]}"

    # Determine early_stopping_flag and max_epochs based on random_label_fraction
    if [ "$random_label_fraction" == "None" ]; then
        early_stopping_flag="--early_stopping"
        max_epochs=800
    else
        early_stopping_flag="--early_stopping"
        max_epochs=800
    fi

    # Log file name
    log_file="logs/gpu_${gpu_id}_run_${i}.log"

    # Run the Python script on the assigned GPU
    echo "GPU $gpu_id: Starting run $i with lr=$lr, batch_size=$batch_size, random_label_fraction=$random_label_fraction, early_stopping_flag=$early_stopping_flag, max_epochs=$max_epochs"
    echo "Architecture: kernel_sizes=$kernel_sizes, strides=$strides, paddings=$paddings, out_channels=$out_channels"
    echo "Optimizer: $optimizer_choice"
    (
        CUDA_VISIBLE_DEVICES=$gpu_id python $SCRIPT_PATH \
            --lr $lr \
            --batch_size $batch_size \
            --random_label_fraction $random_label_fraction \
            $early_stopping_flag \
            --max_epochs $max_epochs \
            --kernel_sizes $kernel_sizes \
            --strides $strides \
            --paddings $paddings \
            --out_channels $out_channels \
            --optimizer_choice $optimizer_choice > $log_file 2>&1
    ) &

    pid=$!
    # Add pid to gpu_pids
    gpu_pids[$gpu_id]="${gpu_pids[$gpu_id]} $pid"
    echo "Downloading data files..."
    sleep 10
done

# Wait for all jobs to finish
for gpu_id in "${gpus[@]}"; do
    pids_str="${gpu_pids[$gpu_id]}"
    pids=($pids_str)
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

echo "All tasks completed."