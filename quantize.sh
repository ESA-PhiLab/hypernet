#!/bin/bash

# Script for quantizing a graph using decent_q tool
# Arguments:
#   1: Path to .json file with input and output node names
#   2: Path to the frozen graph
#   3: Path to the .md5 test data file
#   4: Input shape of the data
#   5: Input function for calibrating network data input
#   6: Batch size
#   7: Output directory
#   8: GPU


INPUT_NODE_NAME=$(jq -r '.input_node' "$1")
OUTPUT_NODE_NAME=$(jq -r '.output_node' "$1")

export INPUT_NODE_NAME
export DATA_PATH=$3
export BATCH_SIZE="$6"

decent_q quantize \
 --input_frozen_graph "$2" \
 --input_nodes "$INPUT_NODE_NAME" \
 --input_shapes "$4" \
 --output_nodes "$OUTPUT_NODE_NAME" \
 --input_fn "$5" \
 --method 1 \
 --gpu "$8" \
 --calib_iter 10 \
 --output_dir "$7" \
