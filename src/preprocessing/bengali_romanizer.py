"""
Bengali to Latin script romanization for improved Rohingya translation.
"""

from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)

class BengaliRomanizer:
    """Handles romanization of Bengali text to Latin script."""
    
    def __init__(self):
        """Initialize the romanizer with custom mappings."""
        # Custom mapping for Bengali-specific sounds
        self.custom_bengali_map = {
            'ঁ': 'n',  # Chandrabindu (nasalization)
            'ৎ': 't',  # Khanda Ta
            'ং': 'ng',  # Anusvara
            'ঃ': 'h',  # Visarga
            'ব': 'b',  # Ensure 'ba' is correctly mapped
            'ভ': 'bh',  # Ensure 'bha' is correctly mapped
            'ল': 'l',   # Ensure 'la' is correctly mapped
            'স': 's',   # Ensure 'sa' is correctly mapped
            'খ': 'kh',  # Ensure 'kha' is correctly mapped
            'ভালবাসি': 'bhalobasi',  # Special case for "love"
        }
        
        # Common Bengali words that need special handling
        self.word_mappings = {
            'বাংলা': 'bangla',  # Special case for "Bangla"
            'ভালো': 'bhalo',    # Special case for "Good"
            'বাসি': 'basi',     # Special case for "live/stay"
            'দুঃখ': 'duhkho',   # Special case for "sorrow"
            'ভালবাসি': 'bhalobasi',  # Special case for "love"
        }
        
        # Common Bengali consonant clusters
        self.consonant_clusters = {
            'ন্দ': 'nd',
            'ঙ্গ': 'ng',
            'স্ক': 'sk',
            'ঃখ': 'hkh',  # Special cluster for visarga + kha
            'লব': 'lob',  # Special cluster for "love"
        }
        
        # Words that should keep their trailing 'a'
        self.keep_trailing_a = {
            'bangla', 'kanda', 'bhalo', 'basi', 'duhkho', 'bhalobasi'
        }
        
    def romanize(self, text: str) -> str:
        """
        Convert Bengali text to romanized form.
        
        Args:
            text: Bengali text to romanize
            
        Returns:
            Romanized text in Latin script
        """
        try:
            # Split text into words
            words = text.split()
            romanized_words = []
            
            for word in words:
                # First pass: handle special word cases
                if word in self.word_mappings:
                    romanized_words.append(self.word_mappings[word])
                    continue
                
                # Second pass: handle consonant clusters
                processed_word = word
                for cluster, replacement in self.consonant_clusters.items():
                    processed_word = processed_word.replace(cluster, replacement)
                
                # Third pass: standard transliteration
                romanized = transliterate(processed_word, sanscript.BENGALI, sanscript.IAST)
                
                # Fourth pass: apply custom mappings
                for bengali, latin in self.custom_bengali_map.items():
                    romanized = romanized.replace(bengali, latin)
                
                # Post-processing for better readability
                romanized = (romanized
                            .replace('ā', 'a')
                            .replace('ī', 'i')
                            .replace('ū', 'u')
                            .replace('ṛ', 'ri')
                            .replace('ḷ', 'li')
                            .replace('ṃ', 'm')
                            .replace('ḥ', 'h')
                            .replace('~', 'n')  # Replace any remaining tildes
                            .replace('v', 'b'))  # Replace 'v' with 'b' for Bengali pronunciation
                
                # Handle trailing 'a'
                if romanized.lower() in self.keep_trailing_a:
                    romanized_words.append(romanized)
                    continue
                    
                # Special handling for words ending in 'kho'
                if romanized.endswith('kho'):
                    romanized_words.append(romanized)
                    continue
                
                # Only remove trailing 'a' if the word is longer than 2 characters
                if len(romanized) > 2 and romanized.endswith('a'):
                    # Keep the 'a' for certain endings
                    if not any(romanized.endswith(ending) for ending in ['ka', 'ta', 'da', 'na', 'la', 'kha']):
                        romanized = romanized[:-1]
                
                romanized_words.append(romanized)
            
            return ' '.join(romanized_words)
            
        except Exception as e:
            logger.error(f"Error in romanization: {str(e)}")
            return text  # Return original text if romanization fails
    
    def romanize_file(self, input_file: str, output_file: str):
        """
        Romanize an entire file of Bengali text.
        
        Args:
            input_file: Path to input file containing Bengali text
            output_file: Path to output file for romanized text
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
            
            romanized_lines = [self.romanize(line) for line in lines]
            
            with open(output_file, 'w', encoding='utf-8') as f_out:
                f_out.writelines(romanized_lines)
                
            logger.info(f"Successfully romanized {len(lines)} lines")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

def romanize_text(text: str) -> str:
    """
    Convenience function to romanize Bengali text.
    
    Args:
        text: Bengali text to romanize
        
    Returns:
        Romanized text in Latin script
    """
    romanizer = BengaliRomanizer()
    return romanizer.romanize(text)

def romanize_file(input_file: str, output_file: str):
    """
    Convenience function to romanize a file containing Bengali text.
    
    Args:
        input_file: Path to input file containing Bengali text
        output_file: Path to output file for romanized text
    """
    romanizer = BengaliRomanizer()
    romanizer.romanize_file(input_file, output_file)
