from unittest.mock import MagicMock, patch
from pathlib import Path
from src.models.inference import RohingyaTranslator

def test_rohingya_translator_nllb_init():
    """Ensure RohingyaTranslator uses NLLB-style codes and Auto classes."""
    # We patch MBart because it's currently hardcoded in the file
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_init, \
         patch("transformers.AutoModelForSeq2SeqLM.from_pretrained") as mock_model_init, \
         patch("transformers.MBart50TokenizerFast.from_pretrained"), \
         patch("transformers.MBartForConditionalGeneration.from_pretrained"):
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_init.return_value = mock_tokenizer
        
        translator = RohingyaTranslator(Path("fake/path"))
        
        # Check if AutoTokenizer and AutoModelForSeq2SeqLM were used
        # Note: If the file is still using MBart, this will fail as expected
        mock_tokenizer_init.assert_called_once()
        mock_model_init.assert_called_once()
        
        # Check if NLLB language codes are used
        assert translator.tokenizer.src_lang == "eng_Latn"
        assert translator.tokenizer.tgt_lang == "rhg_Latn"

def test_translate_uses_forced_bos_token_id():
    """Ensure translate method uses forced_bos_token_id from tokenizer."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_init, \
         patch("transformers.AutoModelForSeq2SeqLM.from_pretrained") as mock_model_init, \
         patch("transformers.MBart50TokenizerFast.from_pretrained"), \
         patch("transformers.MBartForConditionalGeneration.from_pretrained"):
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_init.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model_init.return_value = mock_model
        
        translator = RohingyaTranslator(Path("fake/path"))
        
        # Set up for translate call
        translator.tokenizer.convert_tokens_to_ids.return_value = 12345
        translator.tokenizer.return_value = MagicMock() # Return something that responds to .to()
        translator.tokenizer.return_value.to.return_value = {} # Empty dict for **inputs
        
        translator.translate("test text")
        
        # Verify forced_bos_token_id was passed with correct value from convert_tokens_to_ids
        translator.tokenizer.convert_tokens_to_ids.assert_called_with("rhg_Latn")
        args, kwargs = translator.model.generate.call_args
        assert kwargs["forced_bos_token_id"] == 12345