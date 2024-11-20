#!/bin/bash

# Build the Docker image
docker build -t rohingya-translator-test .

# Create test directories if they don't exist
mkdir -p temp_data/processed

# Run the Docker container with mounted volumes
docker run --rm \
    -v "$(pwd)/configs:/app/configs" \
    -v "$(pwd)/temp_data:/app/temp_data" \
    -v "$(pwd)/src:/app/src" \
    -e CONFIG_PATH=configs/model_config.yaml \
    -e LOCAL_TEST=true \
    rohingya-translator-test python scripts/test_training.py
