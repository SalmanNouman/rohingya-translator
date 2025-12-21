import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import re
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text by removing special characters and extra spaces."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_parallel_texts(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Create parallel texts from the dictionary entries."""
    english_texts = []
    rohingya_texts = []
    
    # Group by English word to handle multiple translations
    grouped = df.groupby('english')
    
    for english, group in tqdm(grouped):
        # Get all Rohingya translations for this English word
        translations = group['rohingya'].tolist()
        
        # Add each translation pair
        for rohingya in translations:
            english_texts.append(normalize_text(english))
            rohingya_texts.append(normalize_text(rohingya))
    
    return english_texts, rohingya_texts

def save_texts(texts: List[str], filepath: Path):
    """Save texts to a file, one per line."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

def main():
    logger = setup_logging()
    
    # Setup paths
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dictionary entries
    logger.info("Loading dictionary entries...")
    df = pd.read_csv(raw_dir / 'dictionary_entries.csv')
    
    # Create parallel texts
    logger.info("Creating parallel texts...")
    english_texts, rohingya_texts = create_parallel_texts(df)
    
    # Split into train, validation, and test sets (80%, 10%, 10%)
    logger.info("Splitting data...")
    
    # First split: 80% train, 20% temp
    eng_train, eng_temp, roh_train, roh_temp = train_test_split(
        english_texts, rohingya_texts, test_size=0.2, random_state=42
    )
    
    # Second split: 10% validation, 10% test (half of temp)
    eng_val, eng_test, roh_val, roh_test = train_test_split(
        eng_temp, roh_temp, test_size=0.5, random_state=42
    )
    
    # Save splits
    logger.info("Saving processed data...")
    
    # Training data
    save_texts(eng_train, processed_dir / 'train.en')
    save_texts(roh_train, processed_dir / 'train.roh')
    
    # Validation data
    save_texts(eng_val, processed_dir / 'val.en')
    save_texts(roh_val, processed_dir / 'val.roh')
    
    # Test data
    save_texts(eng_test, processed_dir / 'test.en')
    save_texts(roh_test, processed_dir / 'test.roh')
    
    # Print statistics
    logger.info("\nData Statistics:")
    logger.info(f"Total pairs: {len(english_texts)}")
    logger.info(f"Training pairs: {len(eng_train)}")
    logger.info(f"Validation pairs: {len(eng_val)}")
    logger.info(f"Test pairs: {len(eng_test)}")
    
    # Save vocabulary statistics
    vocab_stats = {
        'english_vocab_size': len(set(' '.join(english_texts).split())),
        'rohingya_vocab_size': len(set(' '.join(rohingya_texts).split())),
        'english_avg_length': np.mean([len(text.split()) for text in english_texts]),
        'rohingya_avg_length': np.mean([len(text.split()) for text in rohingya_texts])
    }
    
    logger.info("\nVocabulary Statistics:")
    logger.info(f"English vocabulary size: {vocab_stats['english_vocab_size']}")
    logger.info(f"Rohingya vocabulary size: {vocab_stats['rohingya_vocab_size']}")
    logger.info(f"Average English text length: {vocab_stats['english_avg_length']:.2f} words")
    logger.info(f"Average Rohingya text length: {vocab_stats['rohingya_avg_length']:.2f} words")
    
    # Save vocabulary statistics
    pd.DataFrame([vocab_stats]).to_csv(processed_dir / 'vocab_stats.csv', index=False)
    
    logger.info("\nPreprocessing complete! Data saved in data/processed/")

if __name__ == "__main__":
    main()
