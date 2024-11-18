import sys
import os
import logging
import argparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re
import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def extract_word_pairs(text: str) -> list:
    """Extract English-Rohingya word pairs from text."""
    entries = []
    lines = text.split('\n')
    current_letter = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a letter heading (A, B, C, etc.)
        if re.match(r'^[A-Z]$', line):
            current_letter = line
            continue
        
        # Skip lines that don't contain translations
        if not any(sep in line for sep in ['=', '-', ':', '–', '→']):
            continue
            
        try:
            # Try different separators
            for separator in ['=', '-', ':', '–', '→']:
                if separator in line:
                    parts = line.split(separator, 1)
                    if len(parts) == 2:
                        english = parts[0].strip().strip('.')
                        rohingya = parts[1].strip()
                        
                        # Extract word type if present
                        type_match = re.search(r'\((.*?)\)', rohingya)
                        word_type = type_match.group(1).strip() if type_match else 'unknown'
                        
                        # Clean up the translation
                        rohingya = re.sub(r'\s*\(.*?\)', '', rohingya).strip()
                        
                        # Add letter category if available
                        if english and rohingya:
                            entries.append({
                                'english': english,
                                'rohingya': rohingya,
                                'word_type': word_type,
                                'letter': current_letter
                            })
                    break
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error: {str(e)}")
            continue
    
    return entries

def scrape_dictionary(output_file: str, delay: float = 2.0):
    """Scrape the Rohingya dictionary website."""
    logger = setup_logging()
    
    # Setup Chrome options
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_argument('--allow-insecure-localhost')
    
    try:
        logger.info("Starting Chrome WebDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dictionary URL
        url = "https://rohingyadictionary.wordpress.com"
        logger.info(f"Accessing dictionary at: {url}")
        
        try:
            driver.get(url)
            time.sleep(delay)  # Wait for page to load
            
            # Wait for content to load
            content = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "entry-content"))
            )
            
            # Extract all text content
            text_content = content.text
            logger.info("Raw content length: %d characters", len(text_content))
            logger.info("First 500 characters:\n%s", text_content[:500])
            
            # Also try getting HTML content
            html_content = content.get_attribute('innerHTML')
            logger.info("HTML content length: %d characters", len(html_content))
            logger.info("First 500 characters of HTML:\n%s", html_content[:500])
            
            # Process the content
            entries = extract_word_pairs(text_content)
            
            if entries:
                # Convert to DataFrame
                df = pd.DataFrame(entries)
                
                # Sort by letter and then English word
                df = df.sort_values(['letter', 'english'])
                
                # Save to CSV
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                # Print statistics
                logger.info(f"Scraped {len(entries)} word pairs:")
                logger.info(f"- Unique English words: {df['english'].nunique()}")
                logger.info(f"- Unique Rohingya words: {df['rohingya'].nunique()}")
                logger.info(f"- Word types: {sorted(df['word_type'].unique().tolist())}")
                logger.info(f"- Letters covered: {sorted(df['letter'].unique().tolist())}")
                logger.info(f"\nData saved to: {output_file}")
            else:
                logger.warning("No entries were found")
                
        except Exception as e:
            logger.error(f"Error processing dictionary: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
    
    finally:
        driver.quit()
        logger.info("Chrome WebDriver closed")

def main():
    parser = argparse.ArgumentParser(description='Scrape Rohingya-English dictionary')
    parser.add_argument('--output', type=str, default='data/raw/rohingya_english_pairs.csv',
                      help='Output filename (default: data/raw/rohingya_english_pairs.csv)')
    parser.add_argument('--delay', type=float, default=2.0,
                      help='Delay between requests in seconds (default: 2.0)')
    args = parser.parse_args()
    
    scrape_dictionary(args.output, args.delay)

if __name__ == "__main__":
    main()
