import pytest
from unittest.mock import MagicMock, patch
from src.models.huggingface_trainer import GCSCheckpointCallback
from transformers import TrainerControl, TrainerState, TrainingArguments

def test_gcs_callback_on_save_best_model():
    """Verify that GCSCheckpointCallback triggers upload when a new best model is saved."""
    mock_args = MagicMock(spec=TrainingArguments)
    mock_state = MagicMock(spec=TrainerState)
    mock_control = MagicMock(spec=TrainerControl)
    
    # Setup state to indicate a best model was just saved
    mock_state.best_model_checkpoint = "path/to/best_checkpoint"
    
    # We need to simulate the condition where the current save IS the best model.
    # TrainerCallback.on_save is called after saving a checkpoint.
    # The callback logic should check if state.best_model_checkpoint matches the current checkpoint or similar.
    
    callback = GCSCheckpointCallback(
        bucket_name="test-bucket",
        project_id="test-project",
        location="test-location"
    )
    
    with patch("src.models.huggingface_trainer.upload_directory") as mock_upload:
        # Simulate on_save call
        callback.on_save(mock_args, mock_state, mock_control)
        
        # In this first iteration, we just want to ensure it tries to upload SOMETHING if configured.
        # We might need to pass the checkpoint path to on_save or determine it from state.
        mock_upload.assert_called_once()

def test_gcs_callback_on_train_end():
    """Verify that GCSCheckpointCallback does NOT trigger upload at the end of training (handled by script)."""
    mock_args = MagicMock(spec=TrainingArguments)
    mock_state = MagicMock(spec=TrainerState)
    mock_control = MagicMock(spec=TrainerControl)
    
    callback = GCSCheckpointCallback(
        bucket_name="test-bucket",
        project_id="test-project"
    )
    
    with patch("src.models.huggingface_trainer.upload_directory") as mock_upload:
        callback.on_train_end(mock_args, mock_state, mock_control)
        mock_upload.assert_not_called()

@patch("src.models.huggingface_trainer.upload_directory")
@patch("src.models.huggingface_trainer.Seq2SeqTrainer")
@patch("src.models.huggingface_trainer.AutoTokenizer.from_pretrained")
@patch("src.models.huggingface_trainer.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.models.huggingface_trainer.RohingyaHFDataset")
def test_train_model_final_gcs_upload(
    mock_dataset,
    mock_model,
    mock_tokenizer,
    mock_trainer_class,
    mock_upload,
    tmp_path
):
    """Verify that train_model calls upload_directory for the final model."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    from src.models.huggingface_trainer import train_model
    
    # Mock trainer behavior
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    
    # Call train_model with GCS enabled
    train_model(
        data_dir, 
        output_dir, 
        gcs_bucket="test-bucket", 
        gcs_project="test-project"
    )
    
    # Verify final upload was called
    # Once for best model (if triggered by on_save during trainer.train, but we didn't trigger it here)
    # Once for final model at the end of train_model
    mock_upload.assert_called()
    
    # Check that final_model was uploaded
    final_upload_call = [call for call in mock_upload.call_args_list if "final_model" in str(call)]
    assert len(final_upload_call) > 0
