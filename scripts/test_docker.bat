@echo off
REM Build the Docker image
docker build -t rohingya-translator-test .

REM Create test directories if they don't exist
mkdir temp_data\processed 2>nul

REM Run the Docker container with mounted volumes
docker run --rm ^
    -v "%cd%/configs:/app/configs" ^
    -v "%cd%/temp_data:/app/temp_data" ^
    -v "%cd%/src:/app/src" ^
    -e CONFIG_PATH=configs/model_config.yaml ^
    -e LOCAL_TEST=true ^
    rohingya-translator-test python scripts/test_training.py
