FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DeepSpeed with CUDA support
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 \
    pip install --no-cache-dir deepspeed

# Copy source code and cloud configs
COPY src/ src/
COPY configs/cloud/ configs/cloud/
COPY cloud/ cloud/

# Set environment variables for better GPU memory management
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=1

# Default to cloud configuration
ENV CONFIG_PATH=configs/cloud/model_config.yaml

# Command to run training with distributed support
CMD ["python", "-m", "torch.distributed.run", \
     "--nproc_per_node=1", \
     "src/train.py", \
     "--config=${CONFIG_PATH}", \
     "--output_dir=${OUTPUT_DIR}"]
