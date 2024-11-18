# Rohingya Language Translator

This project implements a neural machine translation system for the Rohingya language using PyTorch and Transformers. It uses the mBART-50 model architecture, which has shown strong performance on low-resource languages.

## Project Structure
```
rohingya-translator/
├── data/               # Data storage directory
│   ├── raw/           # Raw parallel corpus
│   └── processed/     # Processed and tokenized data
├── src/               # Source code
│   ├── data/         # Data processing utilities
│   ├── models/       # Model architectures
│   └── utils/        # Helper functions
├── configs/          # Configuration files
├── scripts/          # Training and evaluation scripts
└── notebooks/        # Jupyter notebooks for analysis
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Process dictionary text:
```bash
python scripts/process_dictionary_text.py --input data/raw/dictionary.txt --output data/processed/
```

2. Split data:
```bash
python scripts/process_dictionary.py --input data/processed/dictionary_entries.csv --output data/processed/
```

### Training

1. Configure training parameters in `configs/model_config.yaml`

2. Start training:
```bash
python src/train.py --config configs/model_config.yaml
```

### Translation

Use the trained model for translation:
```bash
python src/model/translate.py --text "Hello, how are you?"
```

## Model Architecture

The translation model is based on the mBART-50 architecture, implemented using PyTorch and Hugging Face Transformers. Key components include:

- Encoder-Decoder architecture with shared vocabulary
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Language-specific tokens for source and target languages

### Training Process

1. Data Processing:
   - Text normalization
   - Tokenization using SentencePiece
   - Train/validation/test splitting

2. Model Training:
   - Teacher forcing
   - Label smoothing
   - Mixed precision training (if GPU available)
   - Gradient accumulation
   - Learning rate scheduling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
