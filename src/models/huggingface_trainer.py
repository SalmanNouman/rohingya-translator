import logging
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from src.data.huggingface_dataset import RohingyaHFDataset, setup_logging

def compute_metrics(tokenizer):
    """Create a compute_metrics function."""
    metric = evaluate.load("sacrebleu")
    
    def _compute(eval_preds):
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
    
    return _compute

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
    max_length: int = 128,
    model_name: str = "facebook/nllb-200-distilled-600M"
):
    """Train the translation model using HuggingFace's Seq2SeqTrainer."""
    logger = setup_logging()
    logger.info("Initializing model training...")
    
    # Initialize tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = RohingyaHFDataset(tokenizer, data_dir, "train", max_length)
    val_dataset = RohingyaHFDataset(tokenizer, data_dir, "val", max_length)
    
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
        report_to=["tensorboard"],  # Log to tensorboard
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
        compute_metrics=compute_metrics(tokenizer)
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    logger.info("Training complete!")
    return trainer  # Return trainer for further use if needed