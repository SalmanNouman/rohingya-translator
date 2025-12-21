import unittest
from pathlib import Path
import torch
from transformers import MBart50TokenizerFast
from src.data.dataset import TranslationDataset, prepare_dataset

class TestTranslationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
        cls.tokenizer.src_lang = "en_XX"
        cls.tokenizer.tgt_lang = "ar_AR" # Using Arabic proxy as per project standard
        
        # Create test data
        cls.test_dir = Path("tests/data")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        # dataset.py prepare_dataset expects path without extension and adds .en and .roh
        cls.data_prefix = cls.test_dir / "train"
        with open(f"{cls.data_prefix}.en", "w", encoding="utf-8") as f:
            f.write("Hello world\nHow are you\n")
        with open(f"{cls.data_prefix}.roh", "w", encoding="utf-8") as f:
            f.write("হ্যালো ওয়ার্ল্ড\nআপনি কেমন আছেন\n") # Using Bengali script as placeholder for Rohingya
    
    def test_prepare_dataset(self):
        """Test prepare_dataset function."""
        dataset = prepare_dataset(
            data_path=str(self.data_prefix),
            max_length=128,
            src_lang="en_XX",
            tgt_lang="ar_AR",
            tokenizer=self.tokenizer
        )
        
        self.assertIsInstance(dataset, TranslationDataset)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.source_texts[0], "Hello world")
        
    def test_dataset_item(self):
        """Test dataset __getitem__."""
        dataset = prepare_dataset(
            data_path=str(self.data_prefix),
            max_length=128,
            src_lang="en_XX",
            tgt_lang="ar_AR",
            tokenizer=self.tokenizer
        )
        
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['labels'], torch.Tensor)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()