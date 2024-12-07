#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main.py"


# Define parameter arrays
train_subset_fractions=(0.25 0.5 0.75 1)
random_label_fractions=(0 0.25 0.5 0.75 1)
noise_image_fractions=(0 0.25 0.5 0.75 1)
seeds=(12 10 24)


# Nested loops to cover all combinations of parameters
for seed in "${seeds[@]}"; do
    if [ $train_subset_fractions == 1 ]; then
        for random_label_fraction in "${random_label_fractions[@]}"; do
            python $SCRIPT_PATH \
                --seed $seed \
                --random_label_fraction $random_label_fraction
        done
        for noise_image_fraction in "${noise_image_fractions[@]}"; do
            python $SCRIPT_PATH \
                --seed $seed \
                --noise_image_fraction $noise_image_fraction
        done
    else
        python $SCRIPT_PATH \
            --seed $seed \
            --train_subset_fraction $train_subset_fraction 
    fi
done

