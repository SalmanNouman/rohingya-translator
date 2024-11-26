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
import json

from models.transformer import RohingyaTranslator
from data.dataset import TranslationDataset, prepare_dataset
from utils.metrics import compute_bleu_score, decode_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
        fp16: bool = False,
        resume_from_checkpoint: Optional[str] = None,
        save_steps: int = 1000,
        logging_steps: int = 100,
        eval_steps: int = 1000
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
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
            
            # Load checkpoint if specified
            if resume_from_checkpoint:
                self._load_checkpoint(resume_from_checkpoint)
                
        except Exception as e:
            logger.error(f"Error setting up optimizer and scheduler: {e}")
            raise

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for model, optimizer, and scheduler."""
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load model
            self.model = RohingyaTranslator.from_pretrained(checkpoint['model_path'])
            self.model.to(self.device)
            
            # Load optimizer and scheduler states
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            # Load training state
            self.global_step = checkpoint['global_step']
            self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
            
            logger.info(f"Resumed training from step {self.global_step}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def save_checkpoint(self, tag: str):
        """Save model checkpoint with error handling."""
        try:
            checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model
            self.model.save_pretrained(str(checkpoint_dir))
            
            # Save training state
            checkpoint = {
                'model_path': str(checkpoint_dir),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss
            }
            
            torch.save(checkpoint, checkpoint_dir / "training_state.pt")
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Upload to GCS if in cloud environment
            if self.output_dir.as_posix().startswith('gs://'):
                self._upload_to_gcs(checkpoint_dir)
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.warning("Continuing training despite checkpoint error")

    def _upload_to_gcs(self, source_dir: Path):
        """Upload checkpoint to Google Cloud Storage."""
        try:
            storage_client = setup_cloud_storage()
            if storage_client:
                bucket_name = self.output_dir.as_posix().split('/')[2]
                bucket = storage_client.bucket(bucket_name)
                
                for filepath in source_dir.glob('**/*'):
                    if filepath.is_file():
                        blob_path = f"models/{source_dir.name}/{filepath.name}"
                        blob = bucket.blob(blob_path)
                        blob.upload_from_filename(filepath)
                logger.info(f"Uploaded checkpoint to GCS: gs://{bucket_name}/models/{source_dir.name}")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            logger.warning("Continuing training despite upload error")

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
            
            # Save best model
            if avg_val_loss < self.best_eval_loss:
                self.best_eval_loss = avg_val_loss
                self.save_checkpoint("best")
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            logger.warning("Continuing training despite validation error")

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
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        self.global_step += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({'loss': loss.item()})
                        
                        # Log every logging_steps
                        if self.global_step % self.logging_steps == 0:
                            logger.info(f"Epoch: {epoch+1}, Step: {self.global_step}, Loss: {loss.item():.4f}")
                            
                        # Save checkpoint every save_steps
                        if self.global_step % self.save_steps == 0:
                            self.save_checkpoint(f"step-{self.global_step}")
                            
                        # Evaluate every eval_steps
                        if self.val_loader and self.global_step % self.eval_steps == 0:
                            self.validate(epoch)
                            self.model.train()  # Resume training mode
                            
                    except Exception as e:
                        logger.error(f"Error during training step: {e}")
                        continue
                
                # Save checkpoint at end of epoch
                self.save_checkpoint(f"epoch-{epoch+1}")
                
                avg_loss = total_loss / len(self.train_loader)
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint("interrupted")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.save_checkpoint("error")
            raise

def setup_cloud_storage():
    """Initialize Google Cloud Storage client if running in cloud environment."""
    try:
        storage_client = storage.Client()
        return storage_client
    except Exception as e:
        logger.warning(f"Not running in cloud environment or missing credentials: {e}")
        return None

def load_config(config_path: str):
    """Load configuration file from local filesystem or Google Cloud Storage."""
    try:
        if config_path.startswith('gs://'):
            # Parse GCS path
            bucket_name = config_path.split('/')[2]
            blob_path = '/'.join(config_path.split('/')[3:])
            
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download and parse config
            config_content = blob.download_as_text()
            config = yaml.safe_load(config_content)
        else:
            # Load local file
            with open(config_path) as f:
                config = yaml.safe_load(f)

        # Convert numeric values in training config
        if 'training' in config:
            numeric_fields = [
                'num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',
                'warmup_steps', 'weight_decay', 'logging_steps', 'eval_steps', 'save_steps',
                'save_total_limit', 'gradient_accumulation_steps', 'max_grad_norm',
                'dataloader_num_workers', 'learning_rate', 'max_split_size_mb'
            ]
            for field in numeric_fields:
                if field in config['training']:
                    try:
                        if isinstance(config['training'][field], str):
                            if 'e' in config['training'][field].lower():
                                config['training'][field] = float(config['training'][field])
                            else:
                                config['training'][field] = int(float(config['training'][field]))
                    except ValueError:
                        pass  # Keep as string if conversion fails

        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def main():
    """Main function with improved error handling and cloud storage support."""
    try:
        # Load config
        config_path = os.getenv("CONFIG_PATH", "configs/local_test_config.yaml")
        config = load_config(config_path)
        
        # Initialize model
        model = RohingyaTranslator(config['model'])
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataset=prepare_dataset(
                config["data"]["train_file"],
                config['data']['max_length'],
                config['model']['src_lang'],
                config['model']['tgt_lang'],
                model.tokenizer
            ),
            val_dataset=prepare_dataset(
                config["data"]["valid_file"],
                config['data']['max_length'],
                config['model']['src_lang'],
                config['model']['tgt_lang'],
                model.tokenizer
            ),
            batch_size=config['training']['per_device_train_batch_size'],
            learning_rate=config['training'].get('learning_rate', 2e-5),
            num_epochs=config['training']['num_train_epochs'],
            warmup_steps=config['training']['warmup_steps'],
            output_dir=os.getenv("OUTPUT_DIR", "outputs"),
            fp16=config['training'].get('fp16', False),
            save_steps=config['training'].get('save_steps', 1000),
            logging_steps=config['training'].get('logging_steps', 100),
            eval_steps=config['training'].get('eval_steps', 1000),
            resume_from_checkpoint=config['training'].get('resume_from_checkpoint', None)
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
