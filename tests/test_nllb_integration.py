import unittest
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.transformer import RohingyaTranslator
import os
import json
import shutil
from pathlib import Path

class TestNLLBIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_name = "facebook/nllb-200-distilled-600M"
        cls.test_dir = Path("tests/test_nllb_model")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # We will test loading from the hub, but for faster local tests 
        # in a real CI we might mock. Here we want to verify real loading.
        
    def test_rohingya_translator_loading(self):
        """Test if RohingyaTranslator loads correctly with NLLB defaults."""
        config = {'max_length': 32}
        translator = RohingyaTranslator(config=config, base_model_name=self.model_name)
        self.assertIsNotNone(translator.model)
        self.assertIsNotNone(translator.tokenizer)
        self.assertEqual(translator.src_lang, "eng_Latn")
        self.assertEqual(translator.tgt_lang, "ben_Beng")

    def test_rohingya_translator_inference(self):
        """Test translation flow using RohingyaTranslator."""
        config = {'max_length': 32}
        translator = RohingyaTranslator(config=config, base_model_name=self.model_name)
        
        input_texts = ["Hello world", "How are you?"]
        translations = translator.translate(input_texts)
        
        self.assertIsInstance(translations, list)
        self.assertEqual(len(translations), 2)
        self.assertIsInstance(translations[0], str)
        self.assertGreater(len(translations[0]), 0)

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()
