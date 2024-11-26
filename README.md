# Rohingya Translator

A machine learning project to translate between English and Rohingya using the mBART-50 model.

## Project Status

Currently in active development with the following milestones:
- Initial project setup and repository structure
- Data preprocessing pipeline
- Model configuration for both local testing and cloud training
- Docker containerization with GPU support
- Cloud training pipeline on Google Vertex AI
- Training in progress (Epoch 6/10)
- Planned: Implement Bengali romanization for improved accuracy

## Project Structure

```
rohingya-translator/
├── cloud/              # Cloud deployment configurations
├── configs/            # Model configurations
│   ├── cloud/         # Cloud-specific configs
│   └── local/         # Local development configs
├── data/              # Dataset directory
│   ├── processed/     # Processed dataset files
│   └── raw/          # Raw dataset files
├── src/               # Source code
└── tests/             # Test files
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For local testing:
- Use configurations in `configs/local/`
- Run `python -m src.train --config_path configs/local/model_config.yaml`

3. For cloud training:
- Build Docker image: `docker build -t gcr.io/airotech-442120/rohingya-translator:v2-memory-opt .`
- Push to GCR: `docker push gcr.io/airotech-442120/rohingya-translator:v2-memory-opt`
- Submit job to Vertex AI using `cloud/vertex_ai_config.yaml`

## Current Challenges & Next Steps

1. Implementing Bengali romanization for improved linguistic accuracy
2. Addressing CUDA environment setup in Docker container
3. Optimizing checkpoint saving mechanism
4. Improving validation logging

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
