import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import evaluate
from tqdm import tqdm

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class RohingyaTranslationDataset:
    def __init__(
        self,
        tokenizer: MBart50TokenizerFast,
        data_dir: Path,
        split: str = "train",
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        
        # Load the data
        self.english_texts = self._load_texts(f"{split}.en")
        self.rohingya_texts = self._load_texts(f"{split}.roh")
        
        # Convert to HuggingFace dataset
        self.dataset = self._create_hf_dataset()
    
    def _load_texts(self, filename: str) -> List[str]:
        """Load texts from a file."""
        with open(self.data_dir / filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def _create_hf_dataset(self) -> HFDataset:
        """Create a HuggingFace dataset."""
        return HFDataset.from_dict({
            'translation': [
                {
                    'en': en,
                    'roh': roh
                }
                for en, roh in zip(self.english_texts, self.rohingya_texts)
            ]
        })
    
    def preprocess_function(self, examples):
        """Preprocess the examples by tokenizing."""
        en_texts = [ex['en'] for ex in examples['translation']]
        roh_texts = [ex['roh'] for ex in examples['translation']]
        
        # Tokenize English inputs
        model_inputs = self.tokenizer(
            en_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize Rohingya targets
        labels = self.tokenizer(
            roh_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def compute_metrics(eval_preds):
    """Compute BLEU score for evaluation."""
    metric = evaluate.load("sacrebleu")
    
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Convert ids to tokens
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    
    return {"bleu": result["score"]}

def train_model(
    data_dir: Path,
    output_dir: Path,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    evaluation_strategy: str = "steps",
    eval_steps: int = 500,
    save_steps: int = 1000,
    max_length: int = 128
):
    """Train the translation model."""
    logger = setup_logging()
    logger.info("Initializing model training...")
    
    # Initialize tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    
    # Add Rohingya language token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<roh>']})
    model.resize_token_embeddings(len(tokenizer))
    
    # Set source and target languages
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "roh_XX"  # Custom token for Rohingya
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = RohingyaTranslationDataset(tokenizer, data_dir, "train", max_length)
    val_dataset = RohingyaTranslationDataset(tokenizer, data_dir, "val", max_length)
    
    # Process datasets
    train_dataset = train_dataset.dataset.map(
        train_dataset.preprocess_function,
        batched=True,
        remove_columns=train_dataset.dataset.column_names
    )
    val_dataset = val_dataset.dataset.map(
        val_dataset.preprocess_function,
        batched=True,
        remove_columns=val_dataset.dataset.column_names
    )
    
    # Initialize trainer
    logger.info("Setting up trainer...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    logger.info("Training complete!")

if __name__ == "__main__":
    # Setup paths
    data_dir = Path("data/processed")
    output_dir = Path("models/rohingya_translator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    train_model(data_dir, output_dir)
