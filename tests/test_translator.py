import unittest
from pathlib import Path
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from src.model.translate import RohingyaTranslator

class TestRohingyaTranslator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create a minimal test model directory
        cls.test_model_dir = Path("tests/test_model")
        cls.test_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a small test model and tokenizer
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
        
        tokenizer.save_pretrained(str(cls.test_model_dir))
        model.save_pretrained(str(cls.test_model_dir))
    
    def test_translator_initialization(self):
        """Test translator initialization."""
        translator = RohingyaTranslator(self.test_model_dir)
        self.assertIsNotNone(translator.model)
        self.assertIsNotNone(translator.tokenizer)
        self.assertEqual(translator.tokenizer.src_lang, "en_XX")
        self.assertEqual(translator.tokenizer.tgt_lang, "roh_XX")
    
    def test_translation(self):
        """Test basic translation functionality."""
        translator = RohingyaTranslator(self.test_model_dir)
        test_text = "Hello, world!"
        
        translation = translator.translate(test_text)
        self.assertIsInstance(translation, str)
        self.assertGreater(len(translation), 0)
    
    def test_translation_with_long_text(self):
        """Test translation with text longer than max_length."""
        translator = RohingyaTranslator(self.test_model_dir)
        long_text = "Hello " * 100
        
        translation = translator.translate(long_text, max_length=128)
        self.assertIsInstance(translation, str)
        self.assertGreater(len(translation), 0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_model_dir)

if __name__ == '__main__':
    unittest.main()
