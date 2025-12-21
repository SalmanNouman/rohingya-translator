import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
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
        save_steps: int = 1000,
        logging_steps: int = 100,
        eval_steps: int = 1000
    ):
        """
        Initializes the Trainer class with model, datasets, and training parameters.
        """
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir # Keep as string for GCS check
        
        # Determine local output directory for saving files first
        if output_dir.startswith("gs://"):
            self.local_output_dir = Path("outputs") # Fallback to local 'outputs'
        else:
            self.local_output_dir = Path(output_dir)
            
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        logger.info(f"Using device: {device}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Local output directory: {self.local_output_dir}")
        logger.info(f"FP16 enabled: {fp16}")
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        ) if val_dataset else None
        
        self.setup_optimizer_and_scheduler()

    def setup_optimizer_and_scheduler(self):
        """Sets up the AdamW optimizer and linear scheduler."""
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=len(self.train_loader) * self.num_epochs
        )

    def save_checkpoint(self, tag: str):
        """Saves a complete model checkpoint."""
        try:
            checkpoint_dir = self.local_output_dir / f"checkpoint-{tag}"
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving checkpoint to {checkpoint_dir}")
            
            # Get unwrapped model
            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Save the model and tokenizer
            unwrapped_model.model.save_pretrained(str(checkpoint_dir))
            unwrapped_model.tokenizer.save_pretrained(str(checkpoint_dir))
            
            # Save optimizer and scheduler states
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_loss': self.best_eval_loss,
            }, checkpoint_dir / "training_state.pt")
            
            # Save training metadata
            metadata = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_loss': self.best_eval_loss,
                'model_type': 'RohingyaTranslator (NLLB-200)',
                'framework': 'pytorch',
                'task': 'translation',
                'source_language': 'rohingya',
                'target_language': 'english'
            }
            
            with open(checkpoint_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Upload to GCS if output_dir is a GCS path
            if self.output_dir.startswith("gs://"):
                self._upload_to_gcs(checkpoint_dir, f"models/checkpoint-{tag}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def save_model(self, tag: str):
        """Saves model weights and tokenizer."""
        try:
            model_dir = self.local_output_dir / f"model-{tag}"
            model_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving model to {model_dir}")
            
            # Get unwrapped model
            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Save using save_pretrained
            unwrapped_model.model.save_pretrained(str(model_dir))
            unwrapped_model.tokenizer.save_pretrained(str(model_dir))
            
            # Save training metadata
            metadata = {
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'model_type': 'RohingyaTranslator',
                'framework': 'pytorch',
                'task': 'translation',
                'source_language': 'rohingya',
                'target_language': 'english'
            }
            
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Upload to GCS if output_dir is a GCS path
            if self.output_dir.startswith("gs://"):
                self._upload_to_gcs(model_dir, f"models/model-{tag}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _upload_to_gcs(self, local_path: Path, gcs_prefix: str):
        """Helper to upload a directory to GCS with detailed logging."""
        try:
            logger.info(f"Uploading {local_path} to GCS at {self.output_dir}/{gcs_prefix}")
            storage_client = storage.Client()
            bucket_name = self.output_dir.split("/")[2]
            bucket = storage_client.bucket(bucket_name)
            
            for local_file in local_path.rglob("*"):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_path)
                    blob_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
                    logger.info(f"Uploading {local_file} to gs://{bucket_name}/{blob_path}")
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(local_file))
            logger.info(f"Successfully uploaded to {self.output_dir}/{gcs_prefix}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}")

    def load_checkpoint(self, tag: str):
        """Loads a complete model checkpoint."""
        try:
            checkpoint_dir = self.local_output_dir / f"checkpoint-{tag}"
            if not checkpoint_dir.exists():
                logger.info(f"No checkpoint found at {checkpoint_dir}")
                return False
                
            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
            unwrapped_model.model = unwrapped_model.model.from_pretrained(str(checkpoint_dir))
            unwrapped_model.tokenizer = unwrapped_model.tokenizer.from_pretrained(str(checkpoint_dir))
            self.model = unwrapped_model.to(self.device)
            
            if (checkpoint_dir / "training_state.pt").exists():
                training_state = torch.load(checkpoint_dir / "training_state.pt")
                self.global_step = training_state['global_step']
                self.current_epoch = training_state['current_epoch']
                self.best_eval_loss = training_state['best_eval_loss']
                self.setup_optimizer_and_scheduler()
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                if self.scheduler and training_state['scheduler_state_dict']:
                    self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                
            logger.info(f"Loaded checkpoint from {checkpoint_dir}")
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def validate(self, epoch: int):
        """Validation step with error handling."""
        try:
            self.model.eval()
            total_val_loss = 0
            num_batches = len(self.val_loader)
            
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()
            
            avg_val_loss = total_val_loss / num_batches
            logger.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < self.best_eval_loss:
                self.best_eval_loss = avg_val_loss
                self.save_model("best")
        except Exception as e:
            logger.warning(f"Error during validation: {e}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                self.model.train()
                total_loss = 0
                
                for step, batch in enumerate(self.train_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    total_loss += loss.item()
                    
                    if self.global_step % self.logging_steps == 0:
                        logger.info(f"Epoch {epoch+1} Step {self.global_step}: Loss = {loss.item():.4f}")
                    
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"step-{self.global_step}")
                
                if self.val_loader:
                    self.validate(epoch)
                self.save_checkpoint(f"epoch-{epoch+1}")
            
            self.save_checkpoint("final")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.save_checkpoint("emergency")
            raise

def load_config(config_path: str) -> dict:
    """Loads YAML configuration from local or GCS."""
    try:
        if config_path.startswith('gs://'):
            bucket_name = config_path.split('/')[2]
            blob_path = '/'.join(config_path.split('/')[3:])
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            config = yaml.safe_load(blob.download_as_text())
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/local_test_config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
        model = RohingyaTranslator(config['model'])
        
        trainer = Trainer(
            model=model,
            train_dataset=prepare_dataset(config["data"]["train_file"], config['data']['max_length'], 
                                         config['model']['src_lang'], config['model']['tgt_lang'], 
                                         model.tokenizer),
            val_dataset=prepare_dataset(config["data"]["valid_file"], config['data']['max_length'], 
                                       config['model']['src_lang'], config['model']['tgt_lang'], 
                                       model.tokenizer),
            batch_size=config['training']['per_device_train_batch_size'],
            learning_rate=float(config['training'].get('learning_rate', 2e-5)),
            num_epochs=config['training']['num_train_epochs'],
            warmup_steps=config['training']['warmup_steps'],
            output_dir=os.getenv("OUTPUT_DIR", "outputs")
        )
        trainer.train()
    except Exception as e:
        logger.error(f"Main failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()