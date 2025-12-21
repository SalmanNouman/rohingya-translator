import logging
from pathlib import Path
from typing import Dict, List
from transformers import MBart50TokenizerFast
from datasets import Dataset as HFDataset

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class RohingyaHFDataset:
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
