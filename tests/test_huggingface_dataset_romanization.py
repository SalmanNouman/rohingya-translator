from unittest.mock import MagicMock, patch, mock_open
import pytest
from src.data.huggingface_dataset import RohingyaHFDataset
from src.preprocessing.bengali_romanizer import BengaliRomanizer

def test_romanizer_initialized():
    """Verify that BengaliRomanizer is initialized within the dataset."""
    mock_tokenizer = MagicMock()
    mock_path = MagicMock()
    
    # Mocking file operations to allow __init__ to complete without real files
    with patch('builtins.open', mock_open(read_data="sample text")), \
         patch('src.data.huggingface_dataset.HFDataset.from_dict') as mock_hf_ds:
        
        dataset = RohingyaHFDataset(mock_tokenizer, mock_path)
        
        # Verify the romanizer attribute is present and is correct type
        assert hasattr(dataset, "romanizer")
        assert dataset.romanizer is not None
        assert isinstance(dataset.romanizer, BengaliRomanizer)

def test_preprocess_calls_romanize():
    """Verify that preprocess_function calls the romanizer on the Rohingya text."""
    # We can skip full init for this test since we mock the romanizer specifically
    dataset = RohingyaHFDataset.__new__(RohingyaHFDataset)
    dataset.tokenizer = MagicMock()
    dataset.max_length = 128
    dataset.romanizer = MagicMock()
    
    examples = {
        'translation': [
            {'en': 'hello', 'roh': 'some_text'}
        ]
    }
    
    # Execute preprocessing
    dataset.preprocess_function(examples)
    
    # Assert that the romanizer was called to process the Rohingya text
    dataset.romanizer.romanize.assert_called()
