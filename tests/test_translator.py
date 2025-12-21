import unittest
from pathlib import Path
import torch
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.transformer import RohingyaTranslator

class TestRohingyaTranslator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create a minimal test model directory
        cls.test_model_dir = Path("tests/test_model")
        cls.test_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a small test model and tokenizer
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        tokenizer.save_pretrained(str(cls.test_model_dir))
        model.save_pretrained(str(cls.test_model_dir))
        
        # Save a dummy config
        config = {
            'src_lang': 'eng_Latn',
            'tgt_lang': 'rhg_Latn',
            'base_model_name': 'facebook/nllb-200-distilled-600M',
            'max_length': 128
        }
        with open(cls.test_model_dir / 'rohingya_translator_config.json', 'w') as f:
            json.dump(config, f)
    
    def test_translator_initialization(self):
        """Test translator initialization."""
        # Use from_pretrained as intended
        translator = RohingyaTranslator.from_pretrained(str(self.test_model_dir))
        self.assertIsNotNone(translator.model)
        self.assertIsNotNone(translator.tokenizer)
        self.assertEqual(translator.src_lang, "eng_Latn")
        self.assertEqual(translator.tgt_lang, "rhg_Latn")
    
    def test_translation(self):
        """Test basic translation functionality."""
        translator = RohingyaTranslator.from_pretrained(str(self.test_model_dir))
        test_text = "Hello, world!"
        
        # Note: translate expects a list of texts in current implementation
        translation = translator.translate([test_text])
        self.assertIsInstance(translation, list)
        self.assertGreater(len(translation), 0)
        self.assertIsInstance(translation[0], str)
    
    def test_translation_with_long_text(self):
        """Test translation with text longer than max_length."""
        translator = RohingyaTranslator.from_pretrained(str(self.test_model_dir))
        long_text = "Hello " * 100
        
        translation = translator.translate([long_text], max_length=128)
        self.assertIsInstance(translation, list)
        self.assertGreater(len(translation), 0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        if cls.test_model_dir.exists():
            shutil.rmtree(cls.test_model_dir)

if __name__ == '__main__':
    unittest.main()