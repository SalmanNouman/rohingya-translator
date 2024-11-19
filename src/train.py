import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional
import yaml
from google.cloud import storage
import argparse

from src.models.transformer import RohingyaTranslator
from src.data.dataset import TranslationDataset, prepare_dataset
from src.utils.metrics import compute_bleu_score, decode_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_cloud_storage():
    """Initialize Google Cloud Storage client if running in cloud environment."""
    try:
        storage_client = storage.Client()
        return storage_client
    except Exception as e:
        logger.warning(f"Not running in cloud environment or missing credentials: {e}")
        return None

def download_from_gcs(gcs_path: str, local_path: str):
    """Download file from Google Cloud Storage."""
    if not gcs_path.startswith('gs://'):
        return gcs_path
    
    try:
        storage_client = setup_cloud_storage()
        if storage_client is None:
            return gcs_path

        bucket_name = gcs_path.split('/')[2]
        blob_path = '/'.join(gcs_path.split('/')[3:])
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {gcs_path} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        return gcs_path

class Trainer:
    def __init__(
        self,
        model: RohingyaTranslator,
        train_dataset: TranslationDataset,
        val_dataset: Optional[TranslationDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs",
        fp16: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16
        
        logger.info(f"Using device: {device}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"FP16 enabled: {fp16}")
        
        # Setup data loaders with error handling
        try:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # Safer for cloud environment
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            ) if val_dataset else None
        except Exception as e:
            logger.error(f"Error setting up data loaders: {e}")
            raise
        
        # Setup optimizer and scheduler
        try:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=len(self.train_loader) * num_epochs
            )
        except Exception as e:
            logger.error(f"Error setting up optimizer and scheduler: {e}")
            raise

    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint with error handling."""
        try:
            checkpoint_dir = self.output_dir / f"checkpoint-{epoch}-{step}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model
            self.model.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Upload to GCS if in cloud environment
            if self.output_dir.as_posix().startswith('gs://'):
                storage_client = setup_cloud_storage()
                if storage_client:
                    bucket_name = self.output_dir.as_posix().split('/')[2]
                    source_dir = checkpoint_dir
                    bucket = storage_client.bucket(bucket_name)
                    
                    for filepath in source_dir.glob('**/*'):
                        if filepath.is_file():
                            blob_path = f"models/checkpoint-{epoch}-{step}/{filepath.name}"
                            blob = bucket.blob(blob_path)
                            blob.upload_from_filename(filepath)
                    logger.info(f"Uploaded checkpoint to GCS: gs://{bucket_name}/models/checkpoint-{epoch}-{step}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def train(self):
        """Main training loop with improved error handling and logging."""
        logger.info("Starting training...")
        try:
            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0
                progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
                
                for step, batch in enumerate(progress_bar):
                    try:
                        # Move batch to device
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        # Backward pass
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # Update progress bar
                        progress_bar.set_postfix({'loss': loss.item()})
                        
                        # Log every 100 steps
                        if step % 100 == 0:
                            logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}")
                            
                        # Save checkpoint every 1000 steps
                        if step > 0 and step % 1000 == 0:
                            self.save_checkpoint(epoch + 1, step)
                            
                    except Exception as e:
                        logger.error(f"Error during training step: {e}")
                        continue
                
                # Validation
                if self.val_loader:
                    self.validate(epoch)
                
                # Save checkpoint at end of epoch
                self.save_checkpoint(epoch + 1, 'final')
                
                avg_loss = total_loss / len(self.train_loader)
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted', 'final')
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def validate(self, epoch: int):
        """Validation step with error handling."""
        try:
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()
            
            avg_val_loss = total_val_loss / len(self.val_loader)
            logger.info(f"Validation loss after epoch {epoch+1}: {avg_val_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise

def main():
    """Main function with improved error handling and cloud storage support."""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train Rohingya Translator')
        parser.add_argument('--config', type=str, required=True, help='Path to config file')
        parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
        parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
        args = parser.parse_args()
        
        # Load and process config
        logger.info(f"Loading config from {args.config}")
        if args.config.startswith('gs://'):
            local_config = '/tmp/model_config.yaml'
            config_path = download_from_gcs(args.config, local_config)
        else:
            config_path = args.config
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Initialize model and datasets
        model = RohingyaTranslator(config['model'])
        
        train_dataset = prepare_dataset(
            f"{args.data_dir}/train",
            config['data']['max_length'],
            config['model']['src_lang'],
            config['model']['tgt_lang']
        )
        
        val_dataset = prepare_dataset(
            f"{args.data_dir}/val",
            config['data']['max_length'],
            config['model']['src_lang'],
            config['model']['tgt_lang']
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config['training']['per_device_train_batch_size'],
            learning_rate=config['training'].get('learning_rate', 2e-5),
            num_epochs=config['training']['num_train_epochs'],
            warmup_steps=config['training']['warmup_steps'],
            output_dir=args.output_dir,
            fp16=config['training'].get('fp16', False)
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
