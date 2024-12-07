#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main.py"


# Define parameter arrays
lr=(0.01 0.005 0.001)
batch_size=(1248 2048 4096)
weight_decay=(0.0001 0.001)

for lr in "${lr[@]}"; do
    for batch_size in "${batch_size[@]}"; do
        for weight_decay in "${weight_decay[@]}"; do
            python $SCRIPT_PATH \
                --lr $lr \
                --batch_size $batch_size \
                --weight_decay $weight_decay
        done
    done
done
