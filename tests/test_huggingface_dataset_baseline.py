import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.data.huggingface_dataset import RohingyaHFDataset
from transformers import PreTrainedTokenizerFast

@pytest.fixture
def mock_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "train.en").write_text("Hello\nWorld", encoding="utf-8")
    (d / "train.roh").write_text("Ola\nMundo", encoding="utf-8")
    return d

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
    # Setup the tokenizer call to return a dictionary with input_ids
    # It needs to handle the call signature in preprocess_function:
    # self.tokenizer(texts, max_length=..., padding=..., truncation=...)
    def side_effect(texts, **kwargs):
        # Return dummy tokens. Length matches number of texts.
        # Each text -> list of ids.
        return {
            "input_ids": [[101, 1, 102] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts]
        }
    tokenizer.side_effect = side_effect
    return tokenizer

def test_load_and_create_hf_dataset(mock_data_dir, mock_tokenizer):
    dataset = RohingyaHFDataset(mock_tokenizer, mock_data_dir, split="train")
    
    # Check if raw data is loaded
    assert len(dataset.english_texts) == 2
    assert dataset.english_texts[0] == "Hello"
    assert dataset.rohingya_texts[0] == "Ola"
    
    # Check HF Dataset structure
    assert len(dataset.dataset) == 2
    assert dataset.dataset[0]['translation']['en'] == "Hello"
    assert dataset.dataset[0]['translation']['roh'] == "Ola"

def test_preprocess_function(mock_data_dir, mock_tokenizer):
    dataset = RohingyaHFDataset(mock_tokenizer, mock_data_dir, split="train")
    
    examples = {
        'translation': [
            {'en': 'Hello', 'roh': 'Ola'},
            {'en': 'World', 'roh': 'Mundo'}
        ]
    }
    
    processed = dataset.preprocess_function(examples)
    
    # Verify tokenizer was called
    assert mock_tokenizer.call_count >= 2 # Once for en, once for roh
    
    # Verify structure of processed data
    assert 'input_ids' in processed
    assert 'labels' in processed
    assert len(processed['input_ids']) == 2
    assert len(processed['labels']) == 2
