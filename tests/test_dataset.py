import unittest
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.data.dataset import TranslationDataset, prepare_dataset

class TestTranslationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Using a small NLLB model for testing
        model_name = "facebook/nllb-200-distilled-600M"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cls.tokenizer.src_lang = "eng_Latn"
        cls.tokenizer.tgt_lang = "rhg_Latn"
        
        # Create test data
        cls.test_dir = Path("tests/data")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        cls.data_prefix = cls.test_dir / "train"
        with open(f"{cls.data_prefix}.en", "w", encoding="utf-8") as f:
            f.write("Hello world\nHow are you\n")
        with open(f"{cls.data_prefix}.roh", "w", encoding="utf-8") as f:
            f.write("হ্যালো ওয়ার্ল্ড\nআপনি কেমন আছেন\n")
    
    def test_prepare_dataset(self):
        """Test prepare_dataset function."""
        dataset = prepare_dataset(
            data_path=str(self.data_prefix),
            max_length=128,
            src_lang="eng_Latn",
            tgt_lang="rhg_Latn",
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
            src_lang="eng_Latn",
            tgt_lang="rhg_Latn",
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
