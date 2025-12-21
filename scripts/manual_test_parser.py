from bs4 import BeautifulSoup
import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.append(src_path)

from data.scraper import WebScraper

# Sample dictionary entries
sample_data = """
.a<>ekkán (n. adj.) ,uggwá (n. adj.) , ek (n. adj.) , eggwá (n. adj.)
.a couple of days<>kessú din (n.) , hoek din (n.) , túra din (n.) , dui tin din (n.)
.a kind of fishing net<>záaiñzal (n.)
.a lot of<>becábicí (adj.) , bóut ziyadá (adj.) , athalaikka (adj.) , bóut bicí (adj.) , ekdóm bicí (adj.) , ebbre bicí (adj.)
.a little bit<>ekkená gori (phrase.)
"""

def test_parser():
    # Create a mock HTML document
    html = f"<div class='entry-content'><p>{sample_data}</p></div>"
    soup = BeautifulSoup(html, 'html.parser')
    
    # Initialize scraper
    scraper = WebScraper(base_url="")
    
    # Extract parallel texts
    results = scraper.extract_parallel_text(soup)
    
    # Print results
    print("\nExtracted word pairs:")
    print("-" * 60)
    for english, rohingya, word_type in results:
        print(f"English: {english}")
        print(f"Rohingya: {rohingya}")
        print(f"Type: {word_type}")
        print("-" * 60)

if __name__ == "__main__":
    test_parser()
