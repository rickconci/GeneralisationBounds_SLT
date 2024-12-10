import itertools
import json

def generate_shell_script(param_dict, script_path="main.py", slots_per_gpu=3, output_script="run_experiments.sh"):
    """
    Generates a shell script to run experiments with all combinations of parameters.

    Args:
        param_dict (dict): Dictionary where keys are parameter names and values are lists of possible values.
        script_path (str): Path to the Python script to execute.
        slots_per_gpu (int): Number of parallel jobs to run per GPU.
        output_script (str): Filename for the generated shell script.
    """
    # Start building the shell script content
    script_lines = [
        "#!/bin/bash",
        "",
        "# Automatically detect available GPUs",
        "gpus=($(nvidia-smi --query-gpu=index --format=csv,noheader))",
        "num_gpus=${#gpus[@]}",
        "",
        "if [ $num_gpus -eq 0 ]; then",
        '    echo "No GPUs found. Exiting."',
        "    exit 1",
        "fi",
        "",
        'echo "Detected GPUs: ${gpus[@]}"',
        'echo "Number of GPUs: $num_gpus"',
        "",
        "# Create a logs directory if it doesn't exist",
        "mkdir -p logs",
        "",
        "# Define the number of slots per GPU",
        f"slots_per_gpu={slots_per_gpu}",
        "",
        "# Initialize an associative array to keep track of PIDs per GPU",
        "declare -A gpu_pids",
        "",
        "for gpu_id in \"${gpus[@]}\"; do",
        '    gpu_pids[$gpu_id]=""',
        "done",
        "",
        "# Initialize an array to keep track of all PIDs",
        "all_pids=()",
        "",
    ]

    # Generate all combinations of parameters
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())
    combinations = list(itertools.product(*param_values))
    num_combinations = len(combinations)

    script_lines.append(f'echo "Number of combinations: {num_combinations}"')
    script_lines.append("")

    # Start the loop over combinations
    script_lines.append("for ((i = 0; i < {} ; i++)); do".format(num_combinations))
    script_lines.append("    while true; do")
    script_lines.append("        assigned_gpu=\"\"")
    script_lines.append("        min_running_jobs=$((slots_per_gpu + 1))")
    script_lines.append("        candidate_gpus=()")
    script_lines.append("        for gpu_id in \"${gpus[@]}\"; do")
    script_lines.append("            # Remove finished PIDs for this GPU")
    script_lines.append("            pids_str=\"${gpu_pids[$gpu_id]}\"")
    script_lines.append("            pids=($pids_str)")
    script_lines.append("            new_pids=()")
    script_lines.append("            for pid in \"${pids[@]}\"; do")
    script_lines.append("                if ps -p \"$pid\" > /dev/null 2>&1; then")
    script_lines.append("                    new_pids+=(\"$pid\")")
    script_lines.append("                fi")
    script_lines.append("            done")
    script_lines.append("            gpu_pids[$gpu_id]=\"${new_pids[@]}\"")
    script_lines.append("            num_running_jobs=${#new_pids[@]}")
    script_lines.append("            if [ \"$num_running_jobs\" -lt \"$slots_per_gpu\" ]; then")
    script_lines.append("                if [ \"$num_running_jobs\" -lt \"$min_running_jobs\" ]; then")
    script_lines.append("                    min_running_jobs=$num_running_jobs")
    script_lines.append("                    candidate_gpus=($gpu_id)")
    script_lines.append("                elif [ \"$num_running_jobs\" -eq \"$min_running_jobs\" ]; then")
    script_lines.append("                    candidate_gpus+=($gpu_id)")
    script_lines.append("                fi")
    script_lines.append("            fi")
    script_lines.append("        done")
    script_lines.append("")
    script_lines.append("        if [ \"${#candidate_gpus[@]}\" -gt 0 ]; then")
    script_lines.append("            # Randomly pick one of the candidate GPUs")
    script_lines.append("            assigned_gpu=${candidate_gpus[$((RANDOM % ${#candidate_gpus[@]}))]}")
    script_lines.append("            break")
    script_lines.append("        else")
    script_lines.append("            # No GPUs available, wait before retrying")
    script_lines.append("            sleep 1")
    script_lines.append("        fi")
    script_lines.append("    done")
    script_lines.append("")

    # Extract parameters for the current combination
    for idx, param_name in enumerate(param_names):
        script_lines.append("    {}=()".format(param_name))
        for value in param_dict[param_name]:
            script_lines.append("    {}+=({})".format(param_name, json.dumps(str(value))))
        script_lines.append("")
    script_lines.append("    # Get parameters for this combination")
    for idx, param_name in enumerate(param_names):
        script_lines.append("    {}_value=${{{}[i]}}".format(param_name, param_name))

    # Build the command to run
    script_lines.append("")
    script_lines.append("    # Log file name")
    script_lines.append('    log_file="logs/gpu_${assigned_gpu}_run_${i}.log"')
    script_lines.append("")
    script_lines.append("    # Run the Python script on the assigned GPU")
    command = [
        "CUDA_VISIBLE_DEVICES=$assigned_gpu python {}".format(script_path)
    ]
    for param_name in param_names:
        command.append("--{} ${{{}_value}}".format(param_name, param_name))
    command_line = " ".join(command) + " > $log_file 2>&1"

    script_lines.append('    echo "GPU $assigned_gpu: Starting run $i with parameters:"')
    param_str = ' '.join(['{}=${{{}_value}}'.format(param_name, param_name) for param_name in param_names])
    script_lines.append('    echo "{}"'.format(param_str))
    script_lines.append("    (")
    script_lines.append("        {}".format(command_line))
    script_lines.append("    ) &")
    script_lines.append("")
    script_lines.append("    pid=$!")
    script_lines.append("    # Add pid to gpu_pids and all_pids")
    script_lines.append('    gpu_pids[$assigned_gpu]="${gpu_pids[$assigned_gpu]} $pid"')
    script_lines.append("    all_pids+=(\"$pid\")")
    script_lines.append("    sleep 1")
    script_lines.append("done")
    script_lines.append("")
    script_lines.append("# Wait for all jobs to finish")
    script_lines.append("for pid in \"${all_pids[@]}\"; do")
    script_lines.append("    wait \"$pid\"")
    script_lines.append("done")
    script_lines.append("")
    script_lines.append('echo "All tasks completed."')

    # Write the script to the output file
    with open(output_script, 'w') as f:
        f.write("\n".join(script_lines))

    # Make the script executable
    import os
    os.chmod(output_script, 0o755)
    print(f"Shell script '{output_script}' has been generated.")


if __name__ == "__main__":
    param_dict = {
        
        "train_subset_fractions": [0.1, 0.25, 0.5, 0.75, 1],
        "random_label_fractions": ["None", 0.5, 1],
        "weight_decay": [0.0, 8e-4],
    }
    generate_shell_script(param_dict, script_path="main.py", slots_per_gpu=2, output_script="SLT_experiment2.sh")