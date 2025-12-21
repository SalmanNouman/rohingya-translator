"""
Test script for the trained Rohingya translator model.
"""

import argparse
import torch
from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_path: Path to the trained model directory
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqGeneration.from_pretrained(model_path)
        logger.info("Successfully loaded model and tokenizer")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def translate(text: str, model, tokenizer, max_length: int = 128):
    """
    Translate text using the loaded model.
    
    Args:
        text: Input text to translate
        model: Loaded translation model
        tokenizer: Loaded tokenizer
        max_length: Maximum length of generated translation
        
    Returns:
        str: Translated text
    """
    try:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate translation
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True
        )
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
        
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test the trained Rohingya translator model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--input_text", type=str, required=True, help="Text to translate")
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(args.model_path)
        
        # Translate input text
        translation = translate(args.input_text, model, tokenizer)
        
        if translation:
            print(f"\nInput text: {args.input_text}")
            print(f"Translation: {translation}\n")
        else:
            print("Translation failed.")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
