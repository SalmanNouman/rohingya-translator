import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.bengali_romanizer import BengaliRomanizer, romanize_text

class TestBengaliRomanizer(unittest.TestCase):
    def setUp(self):
        self.romanizer = BengaliRomanizer()
        
    def test_basic_romanization(self):
        # Test basic Bengali words
        test_cases = [
            ('আমি', 'ami'),  # "I" in Bengali
            ('ভালো', 'bhalo'),  # "Good" in Bengali
            ('আছি', 'achi'),  # "Am" in Bengali
        ]
        
        for bengali, expected in test_cases:
            result = self.romanizer.romanize(bengali)
            self.assertEqual(result.lower(), expected.lower(), 
                           f"Failed to romanize '{bengali}'. Expected '{expected}', got '{result}'")
    
    def test_special_characters(self):
        # Test Bengali special characters
        test_cases = [
            ('কাঁদা', 'kanda'),  # With chandrabindu
            ('বাংলা', 'bangla'),  # With anusvara
            ('দুঃখ', 'duhkho'),  # With visarga
        ]
        
        for bengali, expected in test_cases:
            result = self.romanizer.romanize(bengali)
            self.assertEqual(result.lower(), expected.lower(),
                           f"Failed to handle special character in '{bengali}'. Expected '{expected}', got '{result}'")
    
    def test_convenience_function(self):
        # Test the convenience function
        bengali = 'নমস্কার'  # "Hello" in Bengali
        expected = 'namaskar'
        
        result = romanize_text(bengali)
        self.assertEqual(result.lower(), expected.lower(),
                        f"Convenience function failed. Expected '{expected}', got '{result}'")
    
    def test_sentence(self):
        # Test full sentence romanization
        bengali = 'আমি বাংলা ভালোবাসি'  # "I love Bengali" in Bengali
        expected = 'ami bangla bhalobasi'
        
        result = self.romanizer.romanize(bengali)
        self.assertEqual(result.lower(), expected.lower(),
                        f"Failed to romanize sentence. Expected '{expected}', got '{result}'")

if __name__ == '__main__':
    unittest.main()
