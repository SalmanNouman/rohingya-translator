import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

from train import train

def main():
    """
    Run a small test training job locally to validate the training pipeline
    """
    # Create a small test dataset in temp_data if it doesn't exist
    os.makedirs("temp_data/processed", exist_ok=True)
    
    # Override the config to use local paths and smaller dataset
    os.environ["CONFIG_PATH"] = "configs/model_config.yaml"
    os.environ["LOCAL_TEST"] = "true"
    
    try:
        # Run training with reduced parameters
        train()
        print("✅ Training test completed successfully!")
    except Exception as e:
        print(f"❌ Training test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
