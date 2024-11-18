import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
from typing import List, Optional

from models.transformer import RohingyaTranslator
from data.dataset import TranslationDataset, prepare_dataset
from utils.metrics import compute_bleu_score, decode_predictions

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
        output_dir: str = "outputs"
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(self.train_loader) * num_epochs
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> float:
        """Evaluate the model and return BLEU score."""
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Generate translations
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                    num_beams=4
                )
                
                # Decode predictions and references
                pred_texts = decode_predictions(
                    outputs,
                    self.model.decoder_tokenizer
                )
                ref_texts = decode_predictions(
                    batch["labels"],
                    self.model.decoder_tokenizer
                )
                
                predictions.extend(pred_texts)
                references.extend(ref_texts)
        
        return compute_bleu_score(predictions, references)
    
    def train(self):
        """Train the model for specified number of epochs."""
        best_bleu = 0.0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            avg_loss = self.train_epoch()
            self.logger.info(f"Average loss: {avg_loss:.4f}")
            
            # Evaluate
            if self.val_loader:
                bleu_score = self.evaluate()
                self.logger.info(f"BLEU score: {bleu_score:.4f}")
                
                # Save best model
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    self.save_model("best_model")
            
            # Save checkpoint
            self.save_model(f"checkpoint_epoch_{epoch + 1}")
    
    def save_model(self, name: str):
        """Save model checkpoint."""
        path = self.output_dir / name
        self.model.save_pretrained(path)
        self.logger.info(f"Model saved to {path}")

if __name__ == "__main__":
    # Example usage
    pass
