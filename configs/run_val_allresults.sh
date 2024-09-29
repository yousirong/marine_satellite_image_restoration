#!/bin/bash

# List of data_path values
paths=(
    "model/results/ust21_chl_8day_perfect/600000/10"
    "model/results/ust21_chl_8day_perfect/600000/20"
    "model/results/ust21_chl_8day_perfect/600000/30"
    "model/results/ust21_chl_8day_perfect/600000/40"
    "model/results/ust21_chl_8day_perfect/600000/50"
)

# Path to val.yaml
val_config="configs/val.yaml"

# Loop through each data_path and update val.yaml before running validation
for path in "${paths[@]}"; do
    echo "Running validation for data_path: $path"

    # Use sed or any other method to update data_path in val.yaml
    sed -i "s|data_path:.*|data_path: $path|" $val_config

    # Run the validation command
    python model/run.py --c $val_config --val

    # Optionally, you can sleep between runs to avoid overloading the system
    sleep 2
done
