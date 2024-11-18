import pandas as pd
from pathlib import Path
import re
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean up text by removing extra spaces and unwanted characters."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.replace('_', ' ')  # Replace underscores with spaces
    return text

def process_special_section(line: str, current_section: str) -> dict:
    """Process lines from special sections (days, months)."""
    # Remove numbering at start of line
    line = re.sub(r'^\d+\.\s*', '', line)
    
    # Try different separators
    for sep in ['———————–', '————————', '—————————', '——————', '———————', '<>', '-']:
        if sep in line:
            parts = line.split(sep, 1)
            if len(parts) == 2:
                english = clean_text(parts[0])
                rohingya = clean_text(parts[1])
                
                # Clean up any remaining dashes
                rohingya = re.sub(r'[-—]+', '', rohingya)
                
                # Handle parentheses in Arabic months
                type_match = re.search(r'\((.*?)\)', english)
                if type_match:
                    english = re.sub(r'\s*\(.*?\)', '', english)
                
                return {
                    'english': english.strip(),
                    'rohingya': rohingya.strip(),
                    'word_type': current_section,
                    'category': current_section
                }
    return None

def process_dictionary_text(text: str) -> list:
    """Process dictionary text into structured data."""
    entries = []
    current_section = 'general'
    
    # Split text into lines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.startswith('.DAY & MONTHS'):
            current_section = 'calendar'
            continue
        elif 'ENGLISH vs ROHINGYA' in line:
            if 'Day:' in line:
                current_section = 'day'
            elif 'Month:' in line:
                current_section = 'month_english'
            continue
        elif 'GREGORIAN vs ROHINGYA' in line:
            current_section = 'month_gregorian'
            continue
        elif 'ARABIC vs ROHINGYA' in line:
            current_section = 'month_arabic'
            continue
        elif line.startswith('[') and line.endswith(']'):
            current_section = line.strip('[]').strip().lower().replace(' ', '_')
            continue
            
        try:
            # Handle special sections
            if current_section in ['day', 'month_english', 'month_gregorian', 'month_arabic']:
                entry = process_special_section(line, current_section)
                if entry:
                    entries.append(entry)
                continue
                
            # Regular dictionary entries with <>
            if '<>' in line:
                english_part, rohingya_part = line.split('<>', 1)
                
                # Clean up English part (remove leading dots and clean)
                english = clean_text(english_part.lstrip('.'))
                
                # Split Rohingya translations on commas
                translations = [t.strip() for t in rohingya_part.split(',')]
                
                # Process each translation
                for trans in translations:
                    if not trans:
                        continue
                    
                    # Extract word type from parentheses
                    type_match = re.search(r'\((.*?)\)', trans)
                    word_type = type_match.group(1).strip() if type_match else 'unknown'
                    
                    # Clean up translation (remove word type and clean)
                    rohingya = clean_text(re.sub(r'\s*\(.*?\)', '', trans))
                    
                    if english and rohingya:
                        entries.append({
                            'english': english,
                            'rohingya': rohingya,
                            'word_type': word_type,
                            'category': current_section
                        })
        
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {str(e)}")
            continue
    
    return entries

def main():
    logger = setup_logging()
    
    # Create output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'dictionary_entries.csv'
    
    print("Please paste your dictionary text below.")
    print("Press Ctrl+Z (Windows) or Ctrl+D (Unix) and then Enter when done:")
    
    try:
        # Read all input text
        text = ""
        while True:
            try:
                line = input()
                text += line + "\n"
            except EOFError:
                break
        
        # Process the text
        entries = process_dictionary_text(text)
        
        if entries:
            # Convert to DataFrame
            df = pd.DataFrame(entries)
            
            # Sort by category and English word
            df = df.sort_values(['category', 'english'])
            
            # Remove any duplicate entries
            df = df.drop_duplicates()
            
            # Save to CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            # Print statistics
            logger.info(f"Processed {len(entries)} word pairs:")
            logger.info(f"- Unique English words: {df['english'].nunique()}")
            logger.info(f"- Unique Rohingya words: {df['rohingya'].nunique()}")
            logger.info(f"- Word types: {sorted(df['word_type'].unique().tolist())}")
            logger.info(f"- Categories: {sorted(df['category'].unique().tolist())}")
            
            # Print sample entries from each category
            logger.info("\nSample entries from each category:")
            for category in sorted(df['category'].unique()):
                print(f"\n{category.upper()}:")
                print(df[df['category'] == category].head(2).to_string())
            
            logger.info(f"\nData saved to: {output_file}")
        else:
            logger.warning("No entries were found in the input text")
            
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")

if __name__ == "__main__":
    main()
