import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from src.train import Trainer

class TestTrainerGCS(unittest.TestCase):
    @patch("pathlib.Path.mkdir")
    def setUp(self, mock_mkdir):
        self.mock_model = MagicMock()
        self.mock_train_dataset = MagicMock()
        self.mock_train_dataset.__len__.return_value = 100
        self.mock_val_dataset = MagicMock()
        self.mock_val_dataset.__len__.return_value = 10
        
        # Patching inside setUp to avoid actual initialization issues
        with patch("src.train.Trainer.setup_optimizer_and_scheduler"):
            self.trainer = Trainer(
                model=self.mock_model,
                train_dataset=self.mock_train_dataset,
                val_dataset=self.mock_val_dataset,
                output_dir="gs://test-bucket/outputs"
            )

    @patch("google.cloud.storage.Client")
    def test_save_model_uploads_to_gcs(self, mock_storage_client):
        """Ensure save_model uploads to GCS when output_dir is a gs:// path."""
        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        
        tag = "final"
        # We need to mock Path.mkdir and other file ops to avoid actual IO
        with patch("pathlib.Path.mkdir"), \
             patch("builtins.open", unittest.mock.mock_open()), \
             patch("src.train.logger") as mock_logger:
            
            # Mock rglob to simulate files in model directory
            mock_file = MagicMock(spec=Path)
            mock_file.is_file.return_value = True
            mock_file.relative_to.return_value = Path("model.bin")
            
            # Set up the mock model with model and tokenizer
            self.trainer.model.module = MagicMock()
            self.trainer.model.module.model = MagicMock()
            self.trainer.model.module.tokenizer = MagicMock()
            # If not module
            self.trainer.model.model = MagicMock()
            self.trainer.model.tokenizer = MagicMock()

            with patch("pathlib.Path.rglob", return_value=[mock_file]):
                self.trainer.save_model(tag)
                
                # Check if GCS client was initialized
                mock_storage_client.assert_called_once()
                # Check if bucket was accessed
                mock_storage_client.return_value.bucket.assert_called_with("test-bucket")
                
                # Check if upload was attempted
                mock_bucket.blob.assert_called()

if __name__ == "__main__":
    unittest.main()
