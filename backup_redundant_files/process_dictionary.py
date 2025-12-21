import sys
import os
from pathlib import Path
import re

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.append(src_path)

from data.scraper import WebScraper
import pandas as pd

def process_dictionary_text(text: str) -> list:
    """Process raw dictionary text into structured data."""
    entries = []
    
    # Split text into lines and process each line
    for line in text.split('\n'):
        line = line.strip()
        if not line or not '<>' in line:
            continue
            
        try:
            # Split English and Rohingya parts
            english_part, rohingya_part = line.split('<>', 1)
            english = english_part.strip('.')  # Remove leading dot
            
            # Split Rohingya translations (they're separated by commas)
            translations = rohingya_part.split(',')
            
            # Process each translation
            for trans in translations:
                trans = trans.strip()
                if not trans:
                    continue
                    
                # Extract word type from brackets
                type_match = re.search(r'\((.*?)\)', trans)
                if type_match:
                    word_type = type_match.group(1).strip()
                    # Remove the type from translation
                    translation = re.sub(r'\s*\(.*?\)', '', trans).strip()
                    
                    # Add to entries if both translation and type are valid
                    if translation and word_type:
                        entries.append((english, translation, word_type))
        
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {str(e)}")
            continue
    
    return entries

def main():
    """Process dictionary text and save to CSV."""
    # Get input from user
    print("Please paste your dictionary text (press Ctrl+Z on Windows or Ctrl+D on Unix and then Enter when done):")
    text = sys.stdin.read()
    
    # Process the text
    entries = process_dictionary_text(text)
    
    if not entries:
        print("No entries were found in the input text.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(entries, columns=['english', 'rohingya', 'word_type'])
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_dir / 'dictionary_entries.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Print statistics
    print(f"\nProcessed {len(entries)} dictionary entries:")
    print(f"- Unique English words: {df['english'].nunique()}")
    print(f"- Unique Rohingya words: {df['rohingya'].nunique()}")
    print(f"- Word types: {df['word_type'].unique().tolist()}")
    print(f"\nData saved to: {output_file}")

if __name__ == "__main__":
    main()
