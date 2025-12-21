# Rohingya Translator

A machine learning project to translate between English and Rohingya. The project has recently been modernized to utilize the **NLLB-200** (No Language Left Behind) model, replacing the legacy mBART-50 implementation, to provide superior performance for low-resource languages.

## Project Status

**Current Phase: Modernization & Stabilization (Completed)**

Recent achievements include:
- **Model Migration:** Successfully transitioned from mBART-50 to `facebook/nllb-200-distilled-600M`.
- **Data Normalization:** Integrated a Bengali Romanization pipeline to normalize Rohingya script input.
- **Environment Stabilization:** Fixed Docker and CUDA configurations for reliable GPU-accelerated training on Vertex AI.
- **Cloud Integration:** Implemented robust model checkpointing and uploading to Google Cloud Storage (GCS).

## Project Structure

```
rohingya-translator/
├── cloud/              # Cloud deployment and Vertex AI scripts
├── configs/            # Model and training configurations
│   ├── cloud/         # Cloud-specific configs
│   └── local/         # Local development configs
├── data/              # Dataset directory
├── models/            # Saved models
├── src/               # Source code
│   ├── data/          # Dataset loading and preprocessing
│   ├── inference/     # Inference scripts
│   ├── models/        # Model definitions (NLLB wrapper)
│   └── preprocessing/ # Text normalization (Romanizer)
└── tests/             # Unit and integration tests
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Local Training & Testing
- Use configurations in `configs/local/`
- Run the training loop:
  ```bash
  python -m src.train --config_path configs/local/model_config.yaml
  ```

### 3. Cloud Training (Vertex AI)
- Build the Docker image:
  ```bash
  docker build -t gcr.io/airotech-442120/rohingya-translator:v3-t4-opt .
  ```
- Push to Google Container Registry (GCR):
  ```bash
  docker push gcr.io/airotech-442120/rohingya-translator:v3-t4-opt
  ```
- Submit a job to Vertex AI using the cloud configuration:
  ```bash
  # Ensure you have the google-cloud-aiplatform package installed and authenticated
  python cloud/deploy_to_vertex.py --project-id <YOUR_PROJECT_ID>
  ```

## Key Features

- **NLLB-200 Core:** Utilizes Meta's state-of-the-art multilingual model.
- **Script Normalization:** Automatically handles variations in Rohingya script using Bengali Romanization.
- **Device Awareness:** Automatic detection of CUDA/CPU for inference.
- **Cloud Native:** Built for seamless training on Google Cloud Vertex AI with direct GCS integration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
