FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install git for huggingface
RUN apt-get update && apt-get install -y git

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Create the app directory structure
RUN mkdir -p /app/src

# Set the working directory
WORKDIR /app

# Copy project files
COPY src/ /app/src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the training script
ENTRYPOINT ["python", "-m", "src.train"]
