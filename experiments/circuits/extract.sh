#!/bin/bash
# Extracts a circuit using a given token index
# Example usage: `source experiments/circuits/extract.sh 51`

# Set first positional argument as the shard token idx (default to 0)
SHARD_TOKEN_IDX=${1:-0}

# Set token idx to be the shard token idx % 128
TOKEN_IDX=$((SHARD_TOKEN_IDX % 128))

# Set the sequence idx to be token_idx // 128 * 128
SEQUENCE_IDX=$((SHARD_TOKEN_IDX / 128 * 128))

# Set circuit name
CIRCUIT_NAME="train.0.$SEQUENCE_IDX.$TOKEN_IDX"
echo "Extracting circuit: $CIRCUIT_NAME"

# Extract nodes in parallel
python -m experiments.circuits.nodes --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --layer_idx=0 &
python -m experiments.circuits.nodes --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --layer_idx=1 &
python -m experiments.circuits.nodes --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --layer_idx=2 &
python -m experiments.circuits.nodes --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --layer_idx=3 &
python -m experiments.circuits.nodes --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --layer_idx=4

# Wait for all processes to finish
wait

# Extract edges in parallel
python -m experiments.circuits.edges --circuit=$CIRCUIT_NAME --upstream_layer=0 &
python -m experiments.circuits.edges --circuit=$CIRCUIT_NAME --upstream_layer=1 &
python -m experiments.circuits.edges --circuit=$CIRCUIT_NAME --upstream_layer=2 &
python -m experiments.circuits.edges --circuit=$CIRCUIT_NAME --upstream_layer=3

# Wait for all processes to finish
wait

# Export the circuit
python -m experiments.circuits.mri --circuit=$CIRCUIT_NAME --dirname=mars