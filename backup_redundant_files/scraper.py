import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional
import pandas as pd
import logging
import time
from pathlib import Path
import re

class WebScraper:
    def __init__(
        self,
        base_url: str,
        output_dir: str = "data/raw",
        delay: float = 1.0
    ):
        """
        Web scraper for collecting parallel text data.
        
        Args:
            base_url: Base URL of the website to scrape
            output_dir: Directory to save scraped data
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        
        # Add headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            self.logger.debug(f"Response content length: {len(response.text)}")
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def extract_parallel_text(self, soup: BeautifulSoup) -> List[Tuple[str, str, str]]:
        """
        Extract parallel text from the Rohingya dictionary website.
        Each line contains English word, Rohingya translations, and word types.
        
        Args:
            soup: BeautifulSoup object of the parsed page
            
        Returns:
            List of (english_text, rohingya_text, word_type) tuples
        """
        parallel_texts = []
        
        # Try different ways to find content
        content_elements = (
            soup.find_all(['article', 'div'], class_=['post', 'entry', 'entry-content']) +
            soup.find_all('div', class_=re.compile(r'post|entry|content'))
        )
        
        self.logger.info(f"Found {len(content_elements)} content elements")
        
        for element in content_elements:
            # Get text content
            text = element.get_text('\n', strip=True)
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or not '<>' in line:
                    continue
                
                try:
                    # Split English and Rohingya parts
                    english_part, rohingya_part = line.split('<>', 1)
                    english = english_part.strip('.')  # Remove leading dot if present
                    
                    # Split Rohingya translations (they're separated by commas)
                    translations = []
                    word_types = []
                    
                    # Process each Rohingya translation
                    for trans in rohingya_part.split(','):
                        trans = trans.strip()
                        if not trans:
                            continue
                            
                        # Extract word type from brackets
                        type_match = re.search(r'\((.*?)\)', trans)
                        if type_match:
                            word_type = type_match.group(1).strip()
                            # Remove the type from translation
                            translation = re.sub(r'\s*\(.*?\)', '', trans).strip()
                            
                            # Add to lists if both translation and type are valid
                            if translation and word_type:
                                translations.append(translation)
                                word_types.append(word_type)
                    
                    # Add each translation-type pair as a separate entry
                    for trans, word_type in zip(translations, word_types):
                        if (english, trans, word_type) not in parallel_texts:
                            self.logger.debug(f"Found entry: {english} -> {trans} ({word_type})")
                            parallel_texts.append((english, trans, word_type))
                
                except Exception as e:
                    self.logger.warning(f"Error processing line: {line}")
                    self.logger.warning(f"Error: {str(e)}")
                    continue
        
        self.logger.info(f"Found {len(parallel_texts)} word pairs on this page")
        return parallel_texts
    
    def get_pagination_urls(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract pagination URLs from the page.
        
        Args:
            soup: BeautifulSoup object of the parsed page
            
        Returns:
            List of pagination URLs
        """
        urls = set()  # Use set to avoid duplicates
        
        # Look for links that might be pagination
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            # Check if it's a pagination link
            if any(x in href.lower() for x in ['/page/', '?page=', 'p=', 'paged=']):
                if href.startswith('/'):
                    href = f"https://rohingyadictionary.wordpress.com{href}"
                urls.add(href)
        
        self.logger.info(f"Found {len(urls)} pagination URLs")
        return list(urls)
    
    def save_data(self, data: List[Tuple[str, str, str]], filename: str):
        """
        Save scraped data to CSV file.
        
        Args:
            data: List of (english_text, rohingya_text, word_type) tuples
            filename: Output filename
        """
        df = pd.DataFrame(data, columns=['english', 'rohingya', 'word_type'])
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved {len(data)} parallel texts to {output_path}")
    
    def scrape_all_pages(self, start_url: str, output_filename: str = "rohingya_english_pairs.csv"):
        """
        Scrape all pages starting from the given URL.
        
        Args:
            start_url: URL to start scraping from
            output_filename: Name of output file
        """
        visited_urls = set()
        to_visit = {start_url}
        all_texts = []
        
        while to_visit:
            url = to_visit.pop()
            if url in visited_urls:
                continue
                
            self.logger.info(f"Scraping {url}")
            soup = self.get_page(url)
            
            if soup is not None:
                # Extract parallel texts from current page
                texts = self.extract_parallel_text(soup)
                all_texts.extend(texts)
                
                # Get pagination URLs
                pagination_urls = self.get_pagination_urls(soup)
                for pagination_url in pagination_urls:
                    if pagination_url not in visited_urls:
                        to_visit.add(pagination_url)
                
                visited_urls.add(url)
                
                # Respect the website by waiting between requests
                time.sleep(self.delay)
        
        if all_texts:
            # Remove duplicates while preserving order
            unique_texts = list(dict.fromkeys(all_texts))
            self.save_data(unique_texts, output_filename)
            self.logger.info(f"Total pages scraped: {len(visited_urls)}")
            self.logger.info(f"Total unique word pairs collected: {len(unique_texts)}")
        else:
            self.logger.warning("No data was scraped")
