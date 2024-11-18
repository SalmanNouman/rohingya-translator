import torch
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import PreTrainedTokenizer

class TranslationDataset(Dataset):
    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        """
        Dataset for machine translation.
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts (Rohingya)
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            max_length: Maximum sequence length
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_text = str(self.source_texts[idx])
        target_text = str(self.target_texts[idx])

        # Tokenize source text
        source_encodings = self.source_tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target text
        target_encodings = self.target_tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "labels": target_encodings["input_ids"].squeeze(),
            "decoder_attention_mask": target_encodings["attention_mask"].squeeze()
        }

def prepare_dataset(
    source_texts: List[str],
    target_texts: List[str],
    source_tokenizer: PreTrainedTokenizer,
    target_tokenizer: PreTrainedTokenizer,
    max_length: int = 128
) -> TranslationDataset:
    """
    Prepare a dataset for training or evaluation.
    
    Args:
        source_texts: List of source language texts
        target_texts: List of target language texts
        source_tokenizer: Tokenizer for source language
        target_tokenizer: Tokenizer for target language
        max_length: Maximum sequence length
        
    Returns:
        TranslationDataset instance
    """
    return TranslationDataset(
        source_texts=source_texts,
        target_texts=target_texts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=max_length
    )


import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import re

class RohingyaDataset:
    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        min_freq: int = 2
    ):
        """
        Dataset processor for Rohingya-English parallel texts.
        
        Args:
            data_dir: Directory containing raw data files
            output_dir: Directory to save processed data
            min_freq: Minimum frequency for words to be included in vocabulary
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_freq = min_freq
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize vocabularies
        self.english_vocab = defaultdict(int)
        self.rohingya_vocab = defaultdict(int)
        self.word_types = defaultdict(int)
        
        # Store processed data
        self.parallel_texts: List[Tuple[str, str, str]] = []
    
    def load_data(self, filename: str) -> None:
        """
        Load data from a CSV file containing parallel texts.
        
        Args:
            filename: Name of the CSV file to load
        """
        try:
            df = pd.read_csv(self.data_dir / filename)
            required_columns = ['english', 'rohingya', 'word_type']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Load parallel texts
            self.parallel_texts = list(zip(df['english'], df['rohingya'], df['word_type']))
            self.logger.info(f"Loaded {len(self.parallel_texts)} parallel texts from {filename}")
            
            # Build vocabularies
            self._build_vocabularies()
            
        except Exception as e:
            self.logger.error(f"Error loading data from {filename}: {str(e)}")
            raise
    
    def _build_vocabularies(self) -> None:
        """Build vocabularies from loaded parallel texts."""
        self.english_vocab.clear()
        self.rohingya_vocab.clear()
        self.word_types.clear()
        
        for english, rohingya, word_type in self.parallel_texts:
            # Update English vocabulary
            for word in self._tokenize(english):
                self.english_vocab[word] += 1
            
            # Update Rohingya vocabulary
            for word in self._tokenize(rohingya):
                self.rohingya_vocab[word] += 1
            
            # Update word types
            self.word_types[word_type] += 1
        
        self.logger.info(f"Built vocabularies:")
        self.logger.info(f"- English vocabulary size: {len(self.english_vocab)}")
        self.logger.info(f"- Rohingya vocabulary size: {len(self.rohingya_vocab)}")
        self.logger.info(f"- Word types: {dict(self.word_types)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for vocabulary building.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on whitespace
        text = text.lower()
        
        # Remove punctuation except apostrophes within words
        text = re.sub(r'[^\w\s\']', ' ', text)
        text = re.sub(r'\s\'|\'\s', ' ', text)
        
        return [word for word in text.split() if word]
    
    def get_vocabulary(self, language: str, min_freq: Optional[int] = None) -> Dict[str, int]:
        """
        Get vocabulary for specified language.
        
        Args:
            language: 'english' or 'rohingya'
            min_freq: Minimum frequency for words (defaults to self.min_freq)
            
        Returns:
            Dictionary of words and their frequencies
        """
        if min_freq is None:
            min_freq = self.min_freq
            
        vocab = self.english_vocab if language == 'english' else self.rohingya_vocab
        return {word: freq for word, freq in vocab.items() if freq >= min_freq}
    
    def get_word_types(self) -> Dict[str, int]:
        """Get dictionary of word types and their frequencies."""
        return dict(self.word_types)
    
    def filter_by_word_type(self, word_types: List[str]) -> List[Tuple[str, str, str]]:
        """
        Filter parallel texts by word type.
        
        Args:
            word_types: List of word types to include
            
        Returns:
            Filtered list of parallel texts
        """
        return [
            (eng, roh, wt) for eng, roh, wt in self.parallel_texts
            if wt in word_types
        ]
    
    def save_processed_data(self, filename: str) -> None:
        """
        Save processed parallel texts to CSV.
        
        Args:
            filename: Output filename
        """
        df = pd.DataFrame(
            self.parallel_texts,
            columns=['english', 'rohingya', 'word_type']
        )
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved {len(self.parallel_texts)} processed texts to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_pairs': len(self.parallel_texts),
            'english_vocab_size': len(self.english_vocab),
            'rohingya_vocab_size': len(self.rohingya_vocab),
            'word_types': dict(self.word_types),
            'english_vocab_min_freq': len(self.get_vocabulary('english')),
            'rohingya_vocab_min_freq': len(self.get_vocabulary('rohingya'))
        }
