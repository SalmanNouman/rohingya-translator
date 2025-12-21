import unittest
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.data.dataset import TranslationDataset, prepare_dataset
from src.preprocessing.bengali_romanizer import BengaliRomanizer
import shutil

class TestRomanizerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        cls.tokenizer.src_lang = "eng_Latn"
        
        # Create test data
        cls.test_dir = Path("tests/data_romanizer")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        cls.data_prefix = cls.test_dir / "train"
        with open(f"{cls.data_prefix}.en", "w", encoding="utf-8") as f:
            f.write("Hello world\n")
        with open(f"{cls.data_prefix}.roh", "w", encoding="utf-8") as f:
            # "Hello World" in Bengali script: হ্যালো ওয়ার্ল্ড
            f.write("হ্যালো ওয়ার্ল্ড\n") 

    def test_romanizer_direct(self):
        """Verify Romanizer works as expected independently."""
        romanizer = BengaliRomanizer()
        bengali_text = "হ্যালো ওয়ার্ল্ড"
        # Romanization of "হ্যালো ওয়ার্ল্ড" should produce latin script
        romanized = romanizer.romanize(bengali_text)
        self.assertNotEqual(romanized, bengali_text)
        self.assertRegex(romanized, r'[a-zA-Z]')

    def test_dataset_with_romanizer_enabled(self):
        """Test TranslationDataset with romanizer option enabled."""
        dataset = TranslationDataset(
            source_texts=["Hello world"],
            target_texts=["হ্যালো ওয়ার্ল্ড"],
            tokenizer=self.tokenizer,
            use_romanization=True
        )
        
        # Verify romanizer is initialized
        self.assertIsNotNone(dataset.romanizer)
        
        # Get item and verify it works
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('labels', item)

    def test_dataset_romanization_effect(self):
        """Verify that romanization actually changes the target text before tokenization."""
        # We can mock or just inspect the behavior
        dataset_no_rom = TranslationDataset(
            source_texts=["Hello"],
            target_texts=["হ্যালো"],
            tokenizer=self.tokenizer,
            use_romanization=False
        )
        dataset_rom = TranslationDataset(
            source_texts=["Hello"],
            target_texts=["হ্যালো"],
            tokenizer=self.tokenizer,
            use_romanization=True
        )
        
        # Get labels
        labels_no_rom = dataset_no_rom[0]['labels']
        labels_rom = dataset_rom[0]['labels']
        
        # They should be different because one is tokenized Bengali script, 
        # and the other is tokenized Latin script (romanized)
        self.assertFalse(torch.equal(labels_no_rom, labels_rom))

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()
