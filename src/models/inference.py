from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class RohingyaTranslator:
    def __init__(self, model_path: Path):
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Set source and target languages for NLLB
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "rhg_Latn"
    
    def translate(self, text: str, max_length: int = 128) -> str:
        """Translate English text to Rohingya."""
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate translation
        translated = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
        )
        
        # Decode translation
        translation = self.tokenizer.batch_decode(
            translated,
            skip_special_tokens=True
        )[0]
        
        return translation

def main():
    logger = setup_logging()
    
    # Load model
    model_path = Path("models/rohingya_translator/final_model")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    translator = RohingyaTranslator(model_path)
    
    # Interactive translation loop
    logger.info("Enter English text to translate (or 'q' to quit):")
    while True:
        text = input("> ")
        if text.lower() == 'q':
            break
        
        try:
            translation = translator.translate(text)
            print(f"\nTranslation: {translation}\n")
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")

if __name__ == "__main__":
    main()