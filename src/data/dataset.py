import torch
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import PreTrainedTokenizer
import logging
from collections import defaultdict
from pathlib import Path
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from google.cloud import storage
from src.preprocessing.bengali_romanizer import BengaliRomanizer

class TranslationDataset(Dataset):
    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "rhg_Latn",
        use_romanization: bool = False
    ):
        """
        Dataset for machine translation.
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts (Rohingya)
            tokenizer: NLLB or compatible tokenizer
            max_length: Maximum sequence length
            src_lang: Source language code
            tgt_lang: Target language code
            use_romanization: Whether to apply Bengali Romanization to target texts
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_romanization = use_romanization
        self.romanizer = BengaliRomanizer() if use_romanization else None

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_text = str(self.source_texts[idx])
        target_text = str(self.target_texts[idx])

        # Apply romanization if enabled
        if self.use_romanization and self.romanizer:
            target_text = self.romanizer.romanize(target_text)

        # Tokenize source and target text
        # NLLB uses src_lang and tgt_lang attributes
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        
        # Tokenize source
        source_encodings = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target using text_target
        target_encodings = self.tokenizer(
            text_target=target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension added by tokenizer
        input_ids = source_encodings["input_ids"].squeeze(0)
        attention_mask = source_encodings["attention_mask"].squeeze(0)
        labels = target_encodings["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_dataset(
    data_path: str,
    max_length: int,
    src_lang: str,
    tgt_lang: str,
    tokenizer: PreTrainedTokenizer,
    use_romanization: bool = False
) -> TranslationDataset:
    """
    Prepare a dataset for training or evaluation.
    
    Args:
        data_path: Path to the data directory
        max_length: Maximum sequence length
        src_lang: Source language code
        tgt_lang: Target language code
        tokenizer: NLLB or compatible tokenizer
        use_romanization: Whether to apply Bengali Romanization to target texts
        
    Returns:
        TranslationDataset instance
    """
    # Load the data from GCS or local file system
    if data_path.startswith('gs://'):
        # Download from GCS to temporary file
        storage_client = storage.Client()
        bucket_name = data_path.split('/')[2]
        blob_path = '/'.join(data_path.split('/')[3:])
        bucket = storage_client.bucket(bucket_name)
        
        # Load English data
        en_blob = bucket.blob(f"{blob_path}.en")
        en_content = en_blob.download_as_text()
        source_texts = en_content.strip().split('\n')
        
        # Load Rohingya data
        roh_blob = bucket.blob(f"{blob_path}.roh")
        roh_content = roh_blob.download_as_text()
        target_texts = roh_content.strip().split('\n')
    else:
        # Load from local files
        with open(f"{data_path}.en", 'r', encoding='utf-8') as f:
            source_texts = [line.strip() for line in f]
        with open(f"{data_path}.roh", 'r', encoding='utf-8') as f:
            target_texts = [line.strip() for line in f]
    
    return TranslationDataset(
        source_texts=source_texts,
        target_texts=target_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_romanization=use_romanization
    )

class RohingyaDataset:
    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        min_freq: int = 2
    ):
        """
        Dataset processor for Rohingya-English parallel texts.
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
        """Load data from a CSV file containing parallel texts."""
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
        """Simple tokenization for vocabulary building."""
        text = text.lower()
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r"\s'|'\s", ' ', text)
        return [word for word in text.split() if word]
    
    def get_vocabulary(self, language: str, min_freq: Optional[int] = None) -> Dict[str, int]:
        """Get vocabulary for specified language."""
        if min_freq is None:
            min_freq = self.min_freq
        vocab = self.english_vocab if language == 'english' else self.rohingya_vocab
        return {word: freq for word, freq in vocab.items() if freq >= min_freq}
    
    def get_word_types(self) -> Dict[str, int]:
        """Get dictionary of word types and their frequencies."""
        return dict(self.word_types)
    
    def filter_by_word_type(self, word_types: List[str]) -> List[Tuple[str, str, str]]:
        """Filter parallel texts by word type."""
        return [
            (eng, roh, wt) for eng, roh, wt in self.parallel_texts
            if wt in word_types
        ]
    
    def save_processed_data(self, filename: str) -> None:
        """Save processed parallel texts to CSV."""
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