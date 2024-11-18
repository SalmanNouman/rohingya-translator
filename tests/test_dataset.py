import unittest
from pathlib import Path
import torch
from transformers import MBart50TokenizerFast
from src.model.trainer import RohingyaTranslationDataset

class TestRohingyaTranslationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
        cls.tokenizer.src_lang = "en_XX"
        cls.tokenizer.tgt_lang = "roh_XX"
        
        # Create test data
        cls.test_dir = Path("tests/data")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        with open(cls.test_dir / "train.en", "w", encoding="utf-8") as f:
            f.write("Hello world\\nHow are you\\n")
        with open(cls.test_dir / "train.roh", "w", encoding="utf-8") as f:
            f.write("হ্যালো ওয়ার্ল্ড\\nআপনি কেমন আছেন\\n")
    
    def test_dataset_creation(self):
        """Test dataset creation and basic properties."""
        dataset = RohingyaTranslationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.test_dir,
            split="train",
            max_length=128
        )
        
        self.assertEqual(len(dataset.english_texts), 2)
        self.assertEqual(len(dataset.rohingya_texts), 2)
        self.assertEqual(dataset.english_texts[0], "Hello world")
    
    def test_preprocessing(self):
        """Test preprocessing function."""
        dataset = RohingyaTranslationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.test_dir,
            split="train",
            max_length=128
        )
        
        # Test single example
        example = {
            'translation': [{'en': 'Hello', 'roh': 'হ্যালো'}]
        }
        processed = dataset.preprocess_function(example)
        
        self.assertIn('input_ids', processed)
        self.assertIn('labels', processed)
        self.assertTrue(isinstance(processed['input_ids'], torch.Tensor))
        self.assertTrue(isinstance(processed['labels'], torch.Tensor))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()
