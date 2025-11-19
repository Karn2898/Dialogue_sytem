
import torch
from main.models.encoder import EncoderRNN
from main.models.decoder import LuongAttnDecoderRNN
from main.evaluation.evaluate import GreedySearchDecoder
from main.evaluation.beam_search import BeamSearchDecoder
from main.utils.vocabulary import Vocabulary
from main.utils.load_data import normalizeString
from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MAX_LENGTH


class ChatbotInference:


    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore vocabulary
        self.voc = Vocabulary(self.checkpoint['voc_dict']['name'])
        self.voc.__dict__ = self.checkpoint['voc_dict']

        # Get model parameters
        hidden_size = self.checkpoint.get('hidden_size', 512)
        encoder_n_layers = self.checkpoint.get('encoder_n_layers', 2)
        decoder_n_layers = self.checkpoint.get('decoder_n_layers', 2)

        # Initialize embedding
        self.embedding = torch.nn.Embedding(self.voc.num_words, hidden_size)
        self.embedding.load_state_dict(self.checkpoint['embedding'])

        # Initialize encoder
        self.encoder = EncoderRNN(hidden_size, self.embedding,
                                  encoder_n_layers, dropout=0)
        self.encoder.load_state_dict(self.checkpoint['en'])
        self.encoder = self.encoder.to(device)
        self.encoder.eval()

        # Initialize decoder
        self.decoder = LuongAttnDecoderRNN('dot', self.embedding,
                                           hidden_size, self.voc.num_words,
                                           decoder_n_layers, dropout=0)
        self.decoder.load_state_dict(self.checkpoint['de'])
        self.decoder = self.decoder.to(device)
        self.decoder.eval()

        print(f"Model loaded: {self.voc.num_words} words in vocabulary")

    def generate_response(self, input_text, beam_size=0):
        """Generate response for input text"""
        # Normalize input
        input_text = normalizeString(input_text)

        # Tokenize
        input_tokens = input_text.split()

        # Check length
        if len(input_tokens) > MAX_LENGTH:
            input_tokens = input_tokens[:MAX_LENGTH]

        # Convert to indices
        input_indices = [self.voc.word2index.get(word, PAD_TOKEN)
                         for word in input_tokens]
        input_indices.append(EOS_TOKEN)

        # Create tensor
        input_tensor = torch.LongTensor(input_indices).unsqueeze(1).to(self.device)
        input_length = torch.LongTensor([len(input_indices)]).to(self.device)

        # Choose searcher
        if beam_size > 0:
            searcher = BeamSearchDecoder(self.encoder, self.decoder, beam_size)
            output_indices, _ = searcher(input_tensor, input_length,
                                         MAX_LENGTH, SOS_TOKEN,
                                         EOS_TOKEN, self.device)
        else:
            searcher = GreedySearchDecoder(self.encoder, self.decoder)
            output_indices = searcher(input_tensor, input_length,
                                      MAX_LENGTH, self.device)

        # Convert indices to words
        output_words = [self.voc.index2word[idx] for idx in output_indices
                        if idx not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]

        return ' '.join(output_words)
