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
        
        Args:
            model (RohingyaTranslator): Model instance for training.
            train_dataset (TranslationDataset): Training dataset instance.
            val_dataset (Optional[TranslationDataset]): Validation dataset instance. Defaults to None.
            batch_size (int): Batch size for training. Defaults to 16.
            learning_rate (float): Learning rate for the optimizer. Defaults to 2e-5.
            num_epochs (int): Number of epochs for training. Defaults to 10.
            warmup_steps (int): Number of warmup steps for the scheduler. Defaults to 0.
            device (str): Device for training (cpu or cuda). Defaults to "cuda" if available.
            output_dir (str): Output directory for checkpoints and logs. Defaults to "outputs".
            fp16 (bool): Enable mixed precision training. Defaults to False.
            save_steps (int): Save checkpoint every n steps. Defaults to 1000.
            logging_steps (int): Log training progress every n steps. Defaults to 100.
            eval_steps (int): Evaluate model every n steps. Defaults to 1000.
        """
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
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
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
            self.setup_optimizer_and_scheduler()
        except Exception as e:
            logger.error(f"Error setting up optimizer and scheduler: {e}")
            raise

    def setup_optimizer_and_scheduler(self):
        """
        Sets up the AdamW optimizer and linear scheduler with warmup steps.
        
        This method initializes the optimizer with the model parameters and learning rate,
        and creates a linear scheduler with warmup steps for the training process.
        
        Raises:
            Exception: If there's an error during optimizer or scheduler setup.
        """
        try:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=len(self.train_loader) * self.num_epochs
            )
        except Exception as e:
            logger.error(f"Failed to setup optimizer and scheduler: {e}")
            raise

    def save_checkpoint(self, tag: str):
        """
        Saves a complete model checkpoint including model weights, optimizer state, and training metadata.
        
        Args:
            tag (str): Unique identifier for the checkpoint, typically based on steps or epoch.
            
        The checkpoint includes:
        - Model weights and configuration
        - Tokenizer configuration
        - Optimizer state
        - Scheduler state
        - Training metadata (epoch, step, best loss)
        - Additional metadata about model type and training configuration
        
        If the output directory is a Google Cloud Storage path (starts with 'gs://'),
        the checkpoint will be automatically uploaded to GCS.
        
        Raises:
            Exception: If there's an error during checkpoint saving or GCS upload.
        """
        try:
            checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created checkpoint directory at {checkpoint_dir}")
            
            # Get unwrapped model
            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Save the model and tokenizer
            unwrapped_model.model.save_pretrained(str(checkpoint_dir))
            unwrapped_model.tokenizer.save_pretrained(str(checkpoint_dir))
            logger.info(f"Saved model and tokenizer to {checkpoint_dir}")
            
            # Save optimizer and scheduler states
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_loss': self.best_eval_loss,
            }, checkpoint_dir / "training_state.pt")
            logger.info(f"Saved training state to {checkpoint_dir}/training_state.pt")
            
            # Save training metadata
            metadata = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_loss': self.best_eval_loss,
                'model_type': 'RohingyaTranslator',
                'framework': 'pytorch',
                'task': 'translation',
                'source_language': 'rohingya',
                'target_language': 'english',
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'warmup_steps': self.warmup_steps
            }
            
            with open(checkpoint_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {checkpoint_dir}/metadata.json")
            
            # List all files in checkpoint directory
            logger.info("Files in checkpoint directory:")
            for file_path in checkpoint_dir.rglob("*"):
                if file_path.is_file():
                    logger.info(f"  - {file_path.relative_to(checkpoint_dir)}")
            
            # Upload to GCS if the output directory is a GCS path
            if str(self.output_dir).startswith("gs://"):
                try:
                    logger.info(f"Attempting to upload checkpoint to GCS at {self.output_dir}")
                    storage_client = storage.Client()
                    bucket_name = str(self.output_dir).split("/")[2]
                    logger.info(f"Using bucket: {bucket_name}")
                    bucket = storage_client.bucket(bucket_name)
                    
                    # Upload all files in the checkpoint directory
                    for local_file in checkpoint_dir.rglob("*"):
                        if local_file.is_file():
                            # Get the path relative to the local checkpoint directory
                            relative_path = local_file.relative_to(checkpoint_dir)
                            # Construct the GCS blob path
                            blob_path = f"models/checkpoint-{tag}/{relative_path}".replace("\\", "/")
                            logger.info(f"Uploading {local_file} to gs://{bucket_name}/{blob_path}")
                            blob = bucket.blob(blob_path)
                            blob.upload_from_filename(str(local_file))
                    logger.info(f"Successfully uploaded checkpoint to {self.output_dir}/checkpoint-{tag}")
                except Exception as e:
                    logger.error(f"Failed to upload checkpoint to GCS: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def load_checkpoint(self, tag: str):
        """
        Loads a complete model checkpoint including model weights, optimizer state, and training metadata.
        
        Args:
            tag (str): Identifier of the checkpoint to load.
            
        Returns:
            bool: True if checkpoint was successfully loaded, False if checkpoint not found.
            
        The method loads:
        - Model weights and configuration
        - Tokenizer configuration
        - Optimizer state
        - Scheduler state
        - Training state (epoch, step, best loss)
        
        Raises:
            Exception: If there's an error during checkpoint loading.
        """
        try:
            checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
            if not checkpoint_dir.exists():
                logger.info(f"No checkpoint found at {checkpoint_dir}")
                return False
                
            # Get unwrapped model
            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Load the model and tokenizer
            unwrapped_model.model = unwrapped_model.model.from_pretrained(str(checkpoint_dir))
            unwrapped_model.tokenizer = unwrapped_model.tokenizer.from_pretrained(str(checkpoint_dir))
            self.model = unwrapped_model.to(self.device)
            
            # Load training state
            if (checkpoint_dir / "training_state.pt").exists():
                training_state = torch.load(checkpoint_dir / "training_state.pt")
                self.global_step = training_state['global_step']
                self.current_epoch = training_state['current_epoch']
                self.best_eval_loss = training_state['best_eval_loss']
                
                # Recreate optimizer and scheduler
                self.setup_optimizer_and_scheduler()
                
                # Load their states
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                if self.scheduler and training_state['scheduler_state_dict']:
                    self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                
            logger.info(f"Loaded checkpoint from {checkpoint_dir}")
            logger.info(f"Resuming from epoch {self.current_epoch + 1}, global step {self.global_step}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def save_model(self, tag: str):
        """
        Saves model to GCS.
        
        Args:
            tag (str): Unique identifier for the model, typically based on steps or epoch.
        
        The method saves:
        - Model weights and configuration
        - Tokenizer configuration
        - Training metadata (epoch, step, best loss)
        
        If the output directory is a Google Cloud Storage path (starts with 'gs://'),
        the model will be automatically uploaded to GCS.
        
        Raises:
            Exception: If there's an error during model saving or GCS upload.
        """
        try:
            # Save model locally first
            model_dir = self.output_dir / f"model-{tag}"
            model_dir.mkdir(exist_ok=True, parents=True)
            
            # Get unwrapped model
            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Save the model and tokenizer using save_pretrained
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
                json.dump(metadata, f)
                
            logger.info(f"Saved model to {model_dir}")
            
        except Exception as e:
            logger.warning(f"Error saving model: {e}")
            # Don't raise the error, just log it and continue training
            
    def validate(self, epoch: int):
        """
        Validation step with error handling.
        
        Args:
            epoch (int): Current epoch number.
        
        The method evaluates the model on the validation dataset and logs the validation loss.
        
        Raises:
            Exception: If there's an error during validation.
        """
        try:
            self.model.eval()
            total_val_loss = 0
            num_batches = len(self.val_loader)
            
            logger.info("Starting validation...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()
                    
                    if (batch_idx + 1) % 100 == 0:
                        logger.info(f"Validated {batch_idx + 1}/{num_batches} batches")
            
            avg_val_loss = total_val_loss / num_batches
            logger.info(f"Validation loss after epoch {epoch+1}: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < self.best_eval_loss:
                self.best_eval_loss = avg_val_loss
                self.save_model("best")
            
        except Exception as e:
            logger.warning(f"Error during validation: {e}")
            # Don't raise the error, just log it and continue training

    def train(self):
        """
        Main training loop with improved error handling and checkpoint recovery.
        
        The method trains the model for the specified number of epochs, saving checkpoints periodically.
        
        Raises:
            Exception: If there's an error during training.
        """
        logger.info("Starting training...")
        
        # Try to load latest checkpoint
        latest_epoch = -1
        for checkpoint_dir in self.output_dir.glob("checkpoint-epoch-*"):
            try:
                epoch_num = int(checkpoint_dir.name.split("-")[-1])
                latest_epoch = max(latest_epoch, epoch_num)
            except:
                continue
        
        if latest_epoch >= 0:
            logger.info(f"Found checkpoint for epoch {latest_epoch}")
            self.load_checkpoint(f"epoch-{latest_epoch}")
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                self.model.train()
                total_loss = 0
                num_batches = len(self.train_loader)
                
                for step, batch in enumerate(self.train_loader):
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
                        
                        # Update weights
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        self.global_step += 1
                        
                        # Log training progress
                        if self.global_step % self.logging_steps == 0:
                            avg_loss = total_loss / (step + 1)
                            logger.info(f"Epoch {epoch+1} Step {self.global_step}/{num_batches}: Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}")
                        
                        # Save checkpoint periodically
                        if self.global_step % self.save_steps == 0:
                            self.save_checkpoint(f"step-{self.global_step}")
                            
                    except Exception as batch_e:
                        logger.error(f"Error processing batch: {batch_e}")
                        continue
                
                # Log epoch stats
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed. Average loss: {avg_loss:.4f}")
                
                # Validate after each epoch
                if self.val_loader:
                    self.validate(epoch)
                
                # Save epoch checkpoint
                self.save_checkpoint(f"epoch-{epoch+1}")
            
            # Save final checkpoint
            self.save_checkpoint("final")
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            # Save emergency checkpoint
            self.save_checkpoint("emergency")
            raise

def setup_cloud_storage():
    """
    Initializes Google Cloud Storage client for cloud training environment.
    
    This function:
    1. Checks if running in a cloud environment
    2. Sets up authentication if needed
    3. Initializes the GCS client for checkpoint storage
    
    The function is a no-op if not running in a cloud environment.
    """
    try:
        storage_client = storage.Client()
        return storage_client
    except Exception as e:
        logger.warning(f"Not running in cloud environment or missing credentials: {e}")
        return None

def load_config(config_path: str) -> dict:
    """
    Loads and parses a YAML configuration file from either local filesystem or Google Cloud Storage.
    
    Args:
        config_path (str): Path to the configuration file. Can be local path or GCS path (gs://).
        
    Returns:
        dict: Parsed configuration dictionary containing model and training parameters.
        
    Raises:
        Exception: If the config file cannot be found or parsed.
    """
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
    """
    Main entry point for the training script.
    
    This function:
    1. Parses command line arguments
    2. Loads configuration from YAML file
    3. Sets up cloud storage if needed
    4. Initializes model, datasets, and trainer
    5. Runs the training process
    6. Handles any exceptions during training
    
    Command line arguments:
    --config_path: Path to the YAML configuration file
    """
    parser = argparse.ArgumentParser(description="Rohingya Translator Training Script")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the YAML configuration file")
    args = parser.parse_args()

    try:
        # Load config - prioritize command line arg, then env var, then default
        config_path = args.config_path or os.getenv("CONFIG_PATH", "configs/local_test_config.yaml")
        logger.info(f"Loading configuration from: {config_path}")
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
            eval_steps=config['training'].get('eval_steps', 1000)
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
