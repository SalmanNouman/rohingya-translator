from unittest.mock import MagicMock, patch
from src.models.huggingface_trainer import train_model

def test_train_model_calls_setup_logging():
    """Ensure train_model calls setup_logging."""
    with patch("src.models.huggingface_trainer.setup_logging") as mock_setup_logging:
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # We don't want to actually train, so mock the internal calls
        with patch("transformers.AutoTokenizer.from_pretrained"), \
             patch("transformers.AutoModelForSeq2SeqLM.from_pretrained"), \
             patch("src.models.huggingface_trainer.RohingyaHFDataset"), \
             patch("transformers.Seq2SeqTrainingArguments"), \
             patch("transformers.Seq2SeqTrainer"):
            
            try:
                train_model(MagicMock(), MagicMock())
            except Exception:
                pass
            
            mock_setup_logging.assert_called_once()
            mock_logger.info.assert_any_call("Initializing model training...")
