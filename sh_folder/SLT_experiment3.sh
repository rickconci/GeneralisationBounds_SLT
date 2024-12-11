#!/bin/bash

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

# Define the number of slots per GPU
slots_per_gpu=2

# Initialize an associative array to keep track of PIDs per GPU
declare -A gpu_pids

for gpu_id in "${gpus[@]}"; do
    gpu_pids[$gpu_id]=""
done

# Initialize an array to keep track of all PIDs
all_pids=()

echo "Number of combinations: 96"

for ((i = 0; i < 96 ; i++)); do
    while true; do
        assigned_gpu=""
        min_running_jobs=$((slots_per_gpu + 1))
        candidate_gpus=()
        for gpu_id in "${gpus[@]}"; do
            # Remove finished PIDs for this GPU
            pids_str="${gpu_pids[$gpu_id]}"
            pids=($pids_str)
            new_pids=()
            for pid in "${pids[@]}"; do
                if ps -p "$pid" > /dev/null 2>&1; then
                    new_pids+=("$pid")
                fi
            done
            gpu_pids[$gpu_id]="${new_pids[@]}"
            num_running_jobs=${#new_pids[@]}
            if [ "$num_running_jobs" -lt "$slots_per_gpu" ]; then
                if [ "$num_running_jobs" -lt "$min_running_jobs" ]; then
                    min_running_jobs=$num_running_jobs
                    candidate_gpus=($gpu_id)
                elif [ "$num_running_jobs" -eq "$min_running_jobs" ]; then
                    candidate_gpus+=($gpu_id)
                fi
            fi
        done

        if [ "${#candidate_gpus[@]}" -gt 0 ]; then
            # Randomly pick one of the candidate GPUs
            assigned_gpu=${candidate_gpus[$((RANDOM % ${#candidate_gpus[@]}))]}
            break
        else
            # No GPUs available, wait before retrying
            sleep 1
        fi
    done

    optimizer_choice=()
    optimizer_choice+=("SGD")
    optimizer_choice+=("AdamW")

    random_label_fractions=()
    random_label_fractions+=("None")
    random_label_fractions+=("1")

    weight_decay=()
    weight_decay+=("0.0")
    weight_decay+=("0.003")

    batch_size=()
    batch_size+=("32")
    batch_size+=("256")
    batch_size+=("1024")

    lr=()
    lr+=("0.01")
    lr+=("0.001")

    use_warmup=()
    use_warmup+=("--use_warmup")
    use_warmup+=("")

    architectures=()
    architectures+=("2 2 2|1 1 1|0 0 0|600 600 600")

    # Get parameters for this combination
    optimizer_choice_value=${optimizer_choice[i]}
    random_label_fractions_value=${random_label_fractions[i]}
    weight_decay_value=${weight_decay[i]}
    batch_size_value=${batch_size[i]}
    lr_value=${lr[i]}
    use_warmup_value=${use_warmup[i]}
    architectures_value=${architectures[i]}

    # Log file name
    log_file="logs/gpu_${assigned_gpu}_run_${i}.log"

    # Run the Python script on the assigned GPU
    echo "GPU $assigned_gpu: Starting run $i with parameters:"
    echo "optimizer_choice=${optimizer_choice_value} random_label_fractions=${random_label_fractions_value} weight_decay=${weight_decay_value} batch_size=${batch_size_value} lr=${lr_value} use_warmup=${use_warmup_value} architectures=${architectures_value}"
    (
        CUDA_VISIBLE_DEVICES=$assigned_gpu python main.py --optimizer_choice ${optimizer_choice_value} --random_label_fractions ${random_label_fractions_value} --weight_decay ${weight_decay_value} --batch_size ${batch_size_value} --lr ${lr_value} --use_warmup ${use_warmup_value} --architectures ${architectures_value} > $log_file 2>&1
    ) &

    pid=$!
    # Add pid to gpu_pids and all_pids
    gpu_pids[$assigned_gpu]="${gpu_pids[$assigned_gpu]} $pid"
    all_pids+=("$pid")
    sleep 1
done

# Wait for all jobs to finish
for pid in "${all_pids[@]}"; do
    wait "$pid"
done

echo "All tasks completed."