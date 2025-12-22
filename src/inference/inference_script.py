"""
Test script for the trained Rohingya translator model.
"""

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from src.preprocessing.bengali_romanizer import BengaliRomanizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """
    Load the trained model and tokenizer, and move model to device.
    
    Args:
        model_path: Path to the trained model directory
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device the model is on
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded model and tokenizer on {device}")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def translate(text: str, model, tokenizer, device, max_length: int = 128):
    """
    Translate text using the loaded model.
    
    Args:
        text: Input text to translate
        model: Loaded translation model
        tokenizer: Loaded tokenizer
        device: Device the model is on
        max_length: Maximum length of generated translation
        
    Returns:
        str: Translated text
    """
    try:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
    parser.add_argument("--romanize", action="store_true", help="Apply Bengali romanization to input (use if input contains Bengali script)")
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model(args.model_path)
        
        # Apply romanization if requested (to match training preprocessing)
        input_text = args.input_text
        if args.romanize:
            romanizer = BengaliRomanizer()
            input_text = romanizer.romanize(input_text)
            logger.info(f"Romanized input: {input_text}")
        
        # Translate input text
        translation = translate(input_text, model, tokenizer, device)
        
        if translation:
            print(f"\nInput text: {args.input_text}")
            if args.romanize:
                print(f"Romanized: {input_text}")
            print(f"Translation: {translation}\n")
        else:
            print("Translation failed.")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()