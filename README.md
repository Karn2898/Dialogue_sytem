# Dialogue System using PyTorch

A sequence-to-sequence (seq2seq) dialogue system built with PyTorch, implementing encoder-decoder architecture with attention mechanism for conversational AI applications.

## Features

- **Encoder-Decoder Architecture**: GRU-based seq2seq model for dialogue generation
- **Attention Mechanism**: Improves response quality by focusing on relevant input context
- **Modular Design**: Clean separation of models, training, and utility functions
- **Configurable**: Centralized configuration management for hyperparameters
- **Evaluation Framework**: Built-in evaluation metrics for model performance

## Project Structure

```
Dialogue_sytem/
├── models/              # Neural network architectures
│   ├── encoder.py       # Encoder module
│   ├── decoder.py       # Decoder module
│   └── seq2seq.py       # Seq2seq wrapper
├── Training/            # Training pipeline
│   └── train.py         # Training logic
├── Utils.py/            # Utility functions
│   ├── load.py          # Data loading utilities
│   ├── preprocess.py    # Text preprocessing
│   └── vocabullary.py   # Vocabulary management
├── evaluation/          # Evaluation metrics
├── data/                # Dataset directory
├── config.py/           # Configuration settings
└── main.py/             # Main entry point
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Karn2898/Dialogue_sytem.git
cd Dialogue_sytem
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project supports dialogue datasets in conversational format. Place your dataset in the `data/` directory.

### Expected Format
- Question-answer pairs
- Conversational context
- Text files or CSV format

## Usage

### Training

Run the training script from the main entry point:

```bash
python "main.py/main entry point for traning and testing.py" --mode train
```

### Inference

Generate responses using the trained model:

```bash
python "main.py/main entry point for traning and testing.py" --mode test
```

### Configuration

Modify hyperparameters in `config.py/` directory:
- Learning rate
- Batch size
- Hidden dimensions
- Number of layers
- Dropout rate

## Model Architecture

### Encoder
- Bidirectional GRU layers
- Embedding layer for input tokens
- Hidden state initialization

### Decoder
- GRU layers with attention
- Context vector computation
- Output projection layer

### Attention Mechanism
- Luong attention (dot product)
- Alignment scoring
- Context-aware decoding

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Cross-entropy
- **Gradient Clipping**: Applied for stability
- **Teacher Forcing**: Configurable ratio

## Evaluation

The model is evaluated using:
- Perplexity
- BLEU score
- Response quality metrics

## Examples

```python
# Example usage (to be implemented)
from models.seq2seq import Seq2Seq
from Utils.py.vocabullary import Vocabulary

# Load vocabulary and model
vocab = Vocabulary.load('vocab.pkl')
model = Seq2Seq.load('checkpoint.pth')

# Generate response
input_text = "Hello, how are you?"
response = model.generate_response(input_text)
print(response)
```

## Future Improvements

- [ ] Add pre-trained word embeddings (GloVe, FastText)
- [ ] Implement beam search decoding
- [ ] Add transformer-based architecture option
- [ ] Web interface for interactive conversations
- [ ] Multi-turn dialogue context handling
- [ ] Fine-tuning on domain-specific datasets

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available for educational purposes.

## Author

**Tamaghna Sarkar**
- GitHub: [@Karn2898](https://github.com/Karn2898)
- Email: tamaghna52@gmail.com

## Acknowledgments

- PyTorch documentation and tutorials
- Seq2seq and attention mechanism research papers
- Open-source dialogue system implementations

## References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
