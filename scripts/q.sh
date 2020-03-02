INPUT_NODE_NAME=$(jq '.input_node' ~/Documents/datasets/pavia/freeze_input_output_node_name.json)
OUTPUT_NODE_NAME=$(jq '.output_node' ~/Documents/datasets/pavia/freeze_input_output_node_name.json)
echo $INPUT_NODE_NAME
echo $OUTPUT_NODE_NAME
