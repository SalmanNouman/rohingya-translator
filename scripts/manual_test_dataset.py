import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.append(src_path)

from data.dataset import RohingyaDataset
import pandas as pd

def create_sample_data():
    """Create a sample dataset for testing."""
    data = [
        ('a', 'ekkán', 'n. adj.'),
        ('a', 'uggwá', 'n. adj.'),
        ('a couple of days', 'kessú din', 'n.'),
        ('a couple of days', 'hoek din', 'n.'),
        ('a kind of fishing net', 'záaiñzal', 'n.'),
        ('a lot of', 'becábicí', 'adj.'),
        ('a lot of', 'bóut ziyadá', 'adj.'),
        ('a little bit', 'ekkená gori', 'phrase.')
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['english', 'rohingya', 'word_type'])
    
    # Save to CSV
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / 'sample_data.csv', index=False)
    print("Created sample dataset")

def test_dataset():
    """Test the RohingyaDataset class functionality."""
    # Create sample data
    create_sample_data()
    
    # Initialize dataset
    dataset = RohingyaDataset(min_freq=1)
    
    # Load data
    dataset.load_data('sample_data.csv')
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 60)
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test word type filtering
    print("\nFiltering by word type 'n.':")
    print("-" * 60)
    nouns = dataset.filter_by_word_type(['n.'])
    for eng, roh, wt in nouns:
        print(f"{eng} -> {roh} ({wt})")
    
    # Get vocabularies
    print("\nEnglish Vocabulary:")
    print("-" * 60)
    eng_vocab = dataset.get_vocabulary('english')
    for word, freq in eng_vocab.items():
        print(f"{word}: {freq}")
    
    print("\nRohingya Vocabulary:")
    print("-" * 60)
    roh_vocab = dataset.get_vocabulary('rohingya')
    for word, freq in roh_vocab.items():
        print(f"{word}: {freq}")

if __name__ == "__main__":
    test_dataset()
