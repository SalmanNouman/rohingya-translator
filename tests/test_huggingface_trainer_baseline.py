from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from src.models.huggingface_trainer import train_model

@pytest.fixture
def mock_dirs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return data_dir, output_dir

@patch("src.models.huggingface_trainer.setup_logging")
@patch("src.models.huggingface_trainer.AutoTokenizer.from_pretrained")
@patch("src.models.huggingface_trainer.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.models.huggingface_trainer.RohingyaHFDataset")
@patch("src.models.huggingface_trainer.Seq2SeqTrainingArguments")
@patch("src.models.huggingface_trainer.Seq2SeqTrainer")
@patch("src.models.huggingface_trainer.DataCollatorForSeq2Seq")
def test_train_model_initialization(
    mock_collator,
    mock_trainer_class,
    mock_args_class,
    mock_dataset_class,
    mock_model_from_pretrained,
    mock_tokenizer_from_pretrained,
    mock_setup_logging,
    mock_dirs
):
    data_dir, output_dir = mock_dirs
    
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model
    
    mock_dataset = MagicMock()
    mock_dataset.dataset.map.return_value = MagicMock()
    mock_dataset.dataset.column_names = ["translation"]
    mock_dataset_class.return_value = mock_dataset
    
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    
    # Call the function
    train_model(data_dir, output_dir, num_train_epochs=2)
    
    # Verify tokenizer and model loading
    mock_tokenizer_from_pretrained.assert_called_with("facebook/nllb-200-distilled-600M")
    mock_model_from_pretrained.assert_called_with("facebook/nllb-200-distilled-600M")
    
    # Verify dataset loading for train and val
    assert mock_dataset_class.call_count == 2
    
    # Verify TrainingArguments
    mock_args_class.assert_called_once()
    args_call = mock_args_class.call_args[1]
    assert args_call["num_train_epochs"] == 2
    assert args_call["output_dir"] == str(output_dir)
    
    # Verify Trainer initialization
    mock_trainer_class.assert_called_once()
    
    # Verify training and saving
    mock_trainer.train.assert_called_once()
    mock_trainer.save_model.assert_called_once()
