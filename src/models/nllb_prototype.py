import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import sys
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_nllb():
    """
    Prototype to test NLLB-200 loading and basic functionality.
    """
    model_name = "facebook/nllb-200-distilled-600M"
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test Tokenization with Bengali script (common proxy for Rohingya)
        # Using 'ben_Beng' code for Bengali
        text = "হ্যালো ওয়ার্ল্ড" # Hello World in Bengali script
        tokenizer.src_lang = "ben_Beng"
        inputs = tokenizer(text, return_tensors="pt")
        
        logger.info(f"Tokenized inputs: {inputs}")
        logger.info(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        # Load Model
        # Use low_cpu_mem_usage=True for efficiency
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        logger.info("Model loaded successfully.")
        
        # Forward pass (dummy)
        # Target: English
        target_lang = "eng_Latn"
        # NLLB uses convert_tokens_to_ids for language codes
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        
        # Generate
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id, 
            max_length=30
        )
        
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"Input: {text}")
        logger.info(f"Translation ({target_lang}): {translation}")
        
        # Check model size
        param_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of parameters: {param_size/1e6:.2f} M")
        
        return True
        
    except Exception as e:
        logger.error(f"NLLB Prototype failed: {e}")
        return False
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        gc.collect()

if __name__ == "__main__":
    success = test_nllb()
    sys.exit(0 if success else 1)
