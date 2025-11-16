
import os
import sys
import torch
from config import get_config, Config, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, DEVICE

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vocabulary import Vocabulary
from utils.load_data import loadPrepareData
from models.encoder import EncoderRNN
from models.decoder import LuongAttnDecoderRNN
from training.train import trainModel
from evaluation.evaluate import evaluateModel, evaluateInput
from evaluation.beam_search import BeamSearchDecoder


def initialize_model(voc, config):
    """Initialize encoder and decoder models"""
    print('Building encoder and decoder...')

    # Initialize word embeddings
    embedding = torch.nn.Embedding(voc.num_words, config.hidden_size)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(config.hidden_size, embedding,
                         config.encoder_n_layers, config.dropout)
    decoder = LuongAttnDecoderRNN('dot', embedding, config.hidden_size,
                                  voc.num_words, config.decoder_n_layers,
                                  config.dropout)

    # Move models to device
    encoder = encoder.to(config.device)
    decoder = decoder.to(config.device)

    print('Models built and ready!')
    return encoder, decoder, embedding


def load_checkpoint(filepath, encoder, decoder, embedding):
    """Load saved model checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    print(f'Loading checkpoint from {filepath}...')
    checkpoint = torch.load(filepath)

    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    embedding.load_state_dict(checkpoint['embedding'])

    return checkpoint


def main():
    """Main function to coordinate training or testing"""
    args = get_config()

    # Training mode
    if args.train:
        print("Starting training mode...")

        # Load and prepare data
        corpus_name = os.path.basename(args.train).replace('.txt', '')
        voc, pairs = loadPrepareData(corpus_name, args.train)
        voc.trim(args.min_count if hasattr(args, 'min_count') else 1)

        # Create config
        config = Config(
            hidden_size=args.hidden_size,
            encoder_n_layers=args.layers,
            decoder_n_layers=args.layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_iteration=args.iterations,
            print_every=args.print_every,
            save_every=args.save_every
        )

        # Initialize models
        encoder, decoder, embedding = initialize_model(voc, config)

        # Load checkpoint if specified
        if args.load:
            load_checkpoint(args.load, encoder, decoder, embedding)

        # Train the model
        from training.train import trainModel
        trainModel(encoder, decoder, embedding, voc, pairs, config)

    # Testing mode
    elif args.test:
        print("Starting testing mode...")

        # Load checkpoint
        checkpoint = torch.load(args.test)

        # Restore vocabulary
        voc = Vocabulary(checkpoint['voc_dict']['name'])
        voc.__dict__ = checkpoint['voc_dict']

        # Create config from checkpoint
        config = Config(
            hidden_size=args.hidden_size if args.hidden_size != 512 else checkpoint.get('hidden_size', 512),
            encoder_n_layers=args.layers,
            decoder_n_layers=args.layers,
            dropout=0  # No dropout during inference
        )

        # Initialize models
        encoder, decoder, embedding = initialize_model(voc, config)
        load_checkpoint(args.test, encoder, decoder, embedding)

        # Set to evaluation mode
        encoder.eval()
        decoder.eval()

        # Choose decoder (beam search or greedy)
        if args.beam_size > 0:
            print(f'Using beam search with beam size {args.beam_size}')
            searcher = BeamSearchDecoder(encoder, decoder, beam_width=args.beam_size)
        else:
            print('Using greedy search')
            from evaluation.evaluate import GreedySearchDecoder
            searcher = GreedySearchDecoder(encoder, decoder)

        # Interactive or batch evaluation
        if args.interactive:
            evaluateInput(encoder, decoder, searcher, voc, config)
        elif args.corpus:
            evaluateModel(encoder, decoder, searcher, voc, args.corpus, config)
        else:
            print("Error: Must specify --corpus for batch evaluation or use --interactive")

    else:
        print("Error: Must specify --train for training or --test for testing")
        print("Run with -h for help")


if __name__ == '__main__':
    main()
