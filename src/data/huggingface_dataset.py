import logging
from pathlib import Path
from typing import Dict, List
from transformers import PreTrainedTokenizerFast
from datasets import Dataset as HFDataset
from src.preprocessing.bengali_romanizer import BengaliRomanizer

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
        tokenizer: PreTrainedTokenizerFast,
        data_dir: Path,
        split: str = "train",
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.romanizer = BengaliRomanizer()
        
        # Load the data with validation
        self.english_texts = self._load_texts(f"{split}.en")
        self.rohingya_texts = self._load_texts(f"{split}.roh")
        
        # Validate line counts match
        if len(self.english_texts) != len(self.rohingya_texts):
            raise ValueError(
                f"Data mismatch in {split} split: "
                f"English has {len(self.english_texts)} lines, "
                f"Rohingya has {len(self.rohingya_texts)} lines. "
                f"Both files must have the same number of parallel sentences."
            )
        
        if len(self.english_texts) == 0:
            raise ValueError(f"No data found in {split} split. Check your data files.")
        
        # Convert to HuggingFace dataset
        self.dataset = self._create_hf_dataset()
    
    def _load_texts(self, filename: str) -> List[str]:
        """Load texts from a file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}. "
                f"Ensure your data directory contains {filename}."
            )
        with open(file_path, 'r', encoding='utf-8') as f:
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
        
        # Apply romanization to Rohingya texts
        roh_texts = [self.romanizer.romanize(text) for text in roh_texts]
        
        # Tokenize English inputs
        model_inputs = self.tokenizer(
            en_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize Rohingya targets
        labels = self.tokenizer(
            roh_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs