FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set environment variables for CUDA and Python
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    PYTHONPATH=/app \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Install Python dependencies
# We use --no-cache-dir and skip torch/transformers to keep the optimized container versions
COPY requirements.txt .
RUN grep -vE "torch|transformers" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements_filtered.txt

# Copy source files
COPY src/ src/

# Default command
CMD ["python", "-m", "src.train"]
