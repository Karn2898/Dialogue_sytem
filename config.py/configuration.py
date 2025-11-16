"""
Configuration file for chatbot training and inference
"""
import argparse
import torch

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

# Default hyperparameters
MAX_LENGTH = 15
MIN_COUNT = 1
HIDDEN_SIZE = 512
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
CLIP = 50.0
TEACHER_FORCING_RATIO = 1.0
N_ITERATION = 50000
PRINT_EVERY = 500
SAVE_EVERY = 1000

# Paths
MODEL_SAVE_PATH = './save/model/'
DATA_PATH = './data/'

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config():
    """Parse command line arguments and return configuration"""
    parser = argparse.ArgumentParser(description='Seq2Seq Chatbot')

    # Mode selection
    parser.add_argument('-tr', '--train', type=str, help='Training corpus file path')
    parser.add_argument('-te', '--test', type=str, help='Model file path for testing')
    parser.add_argument('-c', '--corpus', type=str, help='Corpus file for evaluation')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode for manual testing')

    # Model parameters
    parser.add_argument('-hi', '--hidden_size', type=int, default=HIDDEN_SIZE,
                        help='Hidden layer size')
    parser.add_argument('-la', '--layers', type=int, default=ENCODER_N_LAYERS,
                        help='Number of encoder/decoder layers')
    parser.add_argument('-dr', '--dropout', type=float, default=DROPOUT,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('-it', '--iterations', type=int, default=N_ITERATION,
                        help='Number of training iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('-p', '--print_every', type=int, default=PRINT_EVERY,
                        help='Print loss every N iterations')
    parser.add_argument('-s', '--save_every', type=int, default=SAVE_EVERY,
                        help='Save model every N iterations')
    parser.add_argument('-l', '--load', type=str, help='Load saved model checkpoint')

    # Beam search
    parser.add_argument('-be', '--beam_size', type=int, default=0,
                        help='Beam search size (0 for greedy search)')

    args = parser.parse_args()
    return args


class Config:


    def __init__(self, **kwargs):
        # Set default values
        self.max_length = MAX_LENGTH
        self.min_count = MIN_COUNT
        self.hidden_size = HIDDEN_SIZE
        self.encoder_n_layers = ENCODER_N_LAYERS
        self.decoder_n_layers = DECODER_N_LAYERS
        self.dropout = DROPOUT
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.clip = CLIP
        self.teacher_forcing_ratio = TEACHER_FORCING_RATIO
        self.n_iteration = N_ITERATION
        self.print_every = PRINT_EVERY
        self.save_every = SAVE_EVERY
        self.device = DEVICE
        self.model_save_path = MODEL_SAVE_PATH
        self.data_path = DATA_PATH

        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
