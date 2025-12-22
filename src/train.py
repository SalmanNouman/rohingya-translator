"""
Entry point for training the Rohingya-English translation model using Hugging Face Trainer.
"""
import argparse
import os
from pathlib import Path
from src.models.huggingface_trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Rohingya-English Translator")
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_DIR", "data/processed"), help="Path to processed data directory")
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "outputs/model"), help="Path to save model output")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 10)), help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 16)), help="Batch size per device")
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "facebook/nllb-200-distilled-600M"), help="Base model name")
    parser.add_argument("--gcs_bucket", type=str, default=os.getenv("GCS_BUCKET"), help="GCS bucket for uploads")
    parser.add_argument("--gcs_project", type=str, default=os.getenv("GCS_PROJECT"), help="GCS project ID")
    parser.add_argument("--gcs_location", type=str, default=os.getenv("GCS_LOCATION", "us-central1"), help="GCS bucket location")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        model_name=args.model_name,
        gcs_bucket=args.gcs_bucket,
        gcs_project=args.gcs_project,
        gcs_location=args.gcs_location
    )

if __name__ == "__main__":
    main()