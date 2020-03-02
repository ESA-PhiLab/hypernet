#!/bin/bash

# Script for quantizing a graph using decent_q command
# Arguments:
#   1: Path to .json file with input and output node names
#   2: Path to the frozen graph
#   3: Path to the .md5 test data file
#   4: Input shape of the data
#   5: Input function
#   6: Output directory

INPUT_NODE_NAME=$(jq '.input_node' "$1")
OUTPUT_NODE_NAME=$(jq '.output_node' "$1")

export INPUT_NODE_NAME
export DATA_DIR=$3

decent_q quantize \
 --input_frozen_graph "$2" \
 --input_nodes "$INPUT_NODE_NAME" \
 --input_shapes $4 \
 --output_nodes "$OUTPUT_NODE_NAME" \
 --input_fn "$5" \
 --method 1 \
 --gpu 0 \
 --calib_iter 10 \
 --output_dir "$6" \
