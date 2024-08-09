#!/bin/bash

# Purpose of this script is to split data from the feature_store into three
# subsets: train, val and test, and store it into a new folder graph_construction.
# Since, Processing stage also construct graphs so one needs to split data for
# into three different folders for the Edge Labelling stage. One can fix this
# by introducing a new stage Graph Construction (Embedding + Heuristic) stage.

# Define the source directory containing the data
src_dir="./run_quick/feature_store"

# Define the destination directory
dest_dir="./run_quick/graph_construction"

# Create destination subdirectories
train_dir="$dest_dir/train"
val_dir="$dest_dir/val"
test_dir="$dest_dir/test"

mkdir -p "$train_dir"
mkdir -p "$val_dir"
mkdir -p "$test_dir"

# Define the split ratios
train_ratio=0.7
val_ratio=0.15
test_ratio=0.15

# Get the total number of files
total_files=$(ls "$src_dir" | wc -l)

# Calculate the number of files for each split
num_train=$(echo "$total_files * $train_ratio" | bc | awk '{print int($1+0.5)}')
num_val=$(echo "$total_files * $val_ratio" | bc | awk '{print int($1+0.5)}')
num_test=$(echo "$total_files * $test_ratio" | bc | awk '{print int($1+0.5)}')

# List all files and shuffle them
files=($(ls "$src_dir" | shuf))

# Split the files into the respective sets
train_files=("${files[@]:0:$num_train}")
val_files=("${files[@]:$num_train:$num_val}")
test_files=("${files[@]:$((num_train + num_val)):$num_test}")

# Copy the files to the respective directories
for file in "${train_files[@]}"; do
    cp "$src_dir/$file" "$train_dir/"
done

for file in "${val_files[@]}"; do
    cp "$src_dir/$file" "$val_dir/"
done

for file in "${test_files[@]}"; do
    cp "$src_dir/$file" "$test_dir/"
done

echo "Files have been split and copied to $train_dir, $val_dir, and $test_dir in $dest_dir."

