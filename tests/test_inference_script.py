import pytest
from unittest.mock import MagicMock, patch
import torch
from src.inference.inference_script import load_model, translate

def test_load_model_success():
    """Verify load_model loads tokenizer and model correctly."""
    with patch("src.inference.inference_script.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("src.inference.inference_script.AutoModelForSeq2SeqLM.from_pretrained") as mock_model_cls:
        
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        
        model, tokenizer, device = load_model("fake_path")
        
        mock_tokenizer.assert_called_with("fake_path")
        mock_model_cls.assert_called_with("fake_path")
        mock_model.to.assert_called()
        mock_model.eval.assert_called()

def test_translate_success():
    """Verify translate function tokenizes and generates text."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    device = torch.device("cpu")
    
    # Mock tokenizer output
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer.decode.return_value = "Translated Text"
    
    # Mock model generation
    mock_model.generate.return_value = torch.tensor([[4, 5, 6]])
    
    result = translate("Hello", mock_model, mock_tokenizer, device)
    
    assert result == "Translated Text"
    mock_tokenizer.assert_called_with("Hello", return_tensors="pt", padding=True, truncation=True)
    mock_model.generate.assert_called()

def test_translate_error_handling():
    """Verify translate handles errors gracefully."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = Exception("Tokenizer Error")
    
    result = translate("Hello", MagicMock(), mock_tokenizer, MagicMock())
    
    assert result is None
