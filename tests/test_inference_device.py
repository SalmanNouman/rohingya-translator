import torch
from unittest.mock import MagicMock, patch
from src.inference.inference_script import load_model, translate

def test_load_model_device_detection():
    """Ensure load_model detects device and moves model to it."""
    model_path = "fake/path"
    
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_model):
        
        # Test loading
        model, tokenizer, device = load_model(model_path)
        
        # Verify device detection
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device == expected_device
        
        # Verify model.to(device) was called
        mock_model.to.assert_called_with(expected_device)
        # Verify model.eval() was called
        mock_model.eval.assert_called_once()

def test_translate_moves_inputs_to_device():
    """
    This test ensures that the translate function moves input tensors 
    to the same device as the model.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    device = torch.device("cpu")
    
    # Mock tokenizer to return a dict of tensors
    mock_input_ids = MagicMock()
    mock_inputs = {"input_ids": mock_input_ids}
    mock_tokenizer.return_value = mock_inputs
    
    translate("hello", mock_model, mock_tokenizer, device=device)
    
    # Check if input_ids were moved to device
    mock_input_ids.to.assert_called_with(device)