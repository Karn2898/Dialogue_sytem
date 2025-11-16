
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamSearchDecoder(nn.Module):


    def __init__(self, encoder, decoder, beam_width=3):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beam_width = beam_width

    def forward(self, input_seq, input_length, max_length, sos_token, eos_token, device):
        """
        Perform beam search decoding

        Args:
            input_seq: Input sequence tensor
            input_length: Length of input sequence
            max_length: Maximum decode length
            sos_token: Start of sentence token ID
            eos_token: End of sentence token ID
            device: Device to run on (cuda/cpu)

        Returns:
            best_sequence: Best decoded sequence
            best_score: Score of best sequence
        """
        # Encode input
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Initialize beam with SOS token
        sequences = [[sos_token]]
        scores = [0.0]

        for _ in range(max_length):
            all_candidates = []

            for i, seq in enumerate(sequences):
                # Skip if sequence already ended
                if seq[-1] == eos_token:
                    all_candidates.append((scores[i], seq))
                    continue

                # Prepare decoder input
                decoder_input = torch.LongTensor([[seq[-1]]]).to(device)

                # Decode one step
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                # Get top k predictions
                topk_scores, topk_indices = torch.topk(decoder_output, self.beam_width)

                # Create new candidates
                for k in range(self.beam_width):
                    candidate_seq = seq + [topk_indices[0][k].item()]
                    candidate_score = scores[i] + topk_scores[0][k].item()
                    all_candidates.append((candidate_score, candidate_seq))

            # Sort all candidates and keep top beam_width
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = [seq for score, seq in ordered[:self.beam_width]]
            scores = [score for score, seq in ordered[:self.beam_width]]

            # Check if all sequences ended
            if all(seq[-1] == eos_token for seq in sequences):
                break

        # Return best sequence
        best_sequence = sequences[0]
        best_score = scores[0]

        return best_sequence, best_score
