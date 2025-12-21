import subprocess
import sys

def install_requirements():
    """Install required packages for web scraping."""
    requirements = [
        'selenium',
        'pandas',
        'webdriver-manager'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")

if __name__ == "__main__":
    install_requirements()
