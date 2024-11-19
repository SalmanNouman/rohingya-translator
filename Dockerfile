FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install git for huggingface
RUN apt-get update && apt-get install -y git

# Create the app directory structure
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . /app/

# Add the src directory to Python path
ENV PYTHONPATH=/app/src

# Default command (can be overridden)
ENTRYPOINT ["python", "-m", "train"]
