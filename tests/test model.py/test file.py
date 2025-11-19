
import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.models.encoder import EncoderRNN
from main.models.decoder import LuongAttnDecoderRNN, Attn
from main.utils.vocabulary import Vocabulary
from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


class TestEncoderRNN(unittest.TestCase):
    #for encoder

    def setUp(self):

        self.hidden_size = 256
        self.vocab_size = 100
        self.n_layers = 2
        self.batch_size = 4
        self.seq_length = 10

        # Create embedding and encoder
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.encoder = EncoderRNN(
            self.hidden_size,
            self.embedding,
            n_layers=self.n_layers,
            dropout=0.1
        )

    def test_encoder_initialization(self):

        self.assertEqual(self.encoder.hidden_size, self.hidden_size)
        self.assertEqual(self.encoder.n_layers, self.n_layers)
        self.assertIsInstance(self.encoder.gru, nn.GRU)

    def test_encoder_forward_pass_shape(self):

        # Create dummy input
        input_seq = torch.randint(0, self.vocab_size, (self.seq_length, self.batch_size))
        input_lengths = torch.tensor([10, 8, 6, 4])

        # Forward pass
        outputs, hidden = self.encoder(input_seq, input_lengths)

        # Check output shapes
        self.assertEqual(outputs.shape[0], self.seq_length, "Output sequence length incorrect")
        self.assertEqual(outputs.shape[1], self.batch_size, "Output batch size incorrect")
        self.assertEqual(outputs.shape[2], self.hidden_size, "Output hidden size incorrect")

        # Check hidden shape: (n_layers * num_directions, batch, hidden_size)
        # Bidirectional GRU has 2 directions
        expected_hidden_layers = self.n_layers * 2
        self.assertEqual(hidden.shape[0], expected_hidden_layers, "Hidden layers incorrect")
        self.assertEqual(hidden.shape[1], self.batch_size, "Hidden batch size incorrect")
        self.assertEqual(hidden.shape[2], self.hidden_size, "Hidden size incorrect")

    def test_encoder_variable_length_input(self):

        input_seq = torch.randint(0, self.vocab_size, (self.seq_length, self.batch_size))
        input_lengths = torch.tensor([10, 7, 5, 3])  # Variable lengths

        # Should not raise error
        outputs, hidden = self.encoder(input_seq, input_lengths)

        # All outputs should be generated
        self.assertEqual(outputs.shape[0], self.seq_length)

    def test_encoder_gradient_flow(self):

        self.encoder.train()

        input_seq = torch.randint(0, self.vocab_size, (self.seq_length, self.batch_size))
        input_lengths = torch.tensor([10, 8, 6, 4])

        # Forward pass
        outputs, hidden = self.encoder(input_seq, input_lengths)

        # Create dummy loss
        loss = outputs.sum() + hidden.sum()
        loss.backward()

        # Check that parameters have gradients
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")


class TestAttnMechanism(unittest.TestCase):
    #for attention mechanism
    def setUp(self):
        """Set up test attention module"""
        self.hidden_size = 256
        self.batch_size = 4
        self.seq_length = 10

    def test_dot_attention(self):

        attn = Attn('dot', self.hidden_size)

        hidden = torch.randn(1, self.batch_size, self.hidden_size)
        encoder_outputs = torch.randn(self.seq_length, self.batch_size, self.hidden_size)

        attn_weights = attn(hidden, encoder_outputs)

        # Check shape
        self.assertEqual(attn_weights.shape, (self.batch_size, 1, self.seq_length))

        # Check attention weights sum to 1
        weights_sum = attn_weights.sum(dim=2)
        self.assertTrue(torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5))

    def test_general_attention(self):

        attn = Attn('general', self.hidden_size)

        hidden = torch.randn(1, self.batch_size, self.hidden_size)
        encoder_outputs = torch.randn(self.seq_length, self.batch_size, self.hidden_size)

        attn_weights = attn(hidden, encoder_outputs)

        # Check shape and normalization
        self.assertEqual(attn_weights.shape, (self.batch_size, 1, self.seq_length))
        weights_sum = attn_weights.sum(dim=2)
        self.assertTrue(torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5))

    def test_concat_attention(self):

        attn = Attn('concat', self.hidden_size)

        hidden = torch.randn(1, self.batch_size, self.hidden_size)
        encoder_outputs = torch.randn(self.seq_length, self.batch_size, self.hidden_size)

        attn_weights = attn(hidden, encoder_outputs)

        # Check shape and normalization
        self.assertEqual(attn_weights.shape, (self.batch_size, 1, self.seq_length))
        weights_sum = attn_weights.sum(dim=2)
        self.assertTrue(torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5))

    def test_invalid_attention_method(self):

        with self.assertRaises(ValueError):
            Attn('invalid_method', self.hidden_size)


class TestLuongAttnDecoderRNN(unittest.TestCase):


    def setUp(self):

        self.hidden_size = 256
        self.vocab_size = 100
        self.n_layers = 2
        self.batch_size = 4
        self.seq_length = 10

        # Create embedding and decoder
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder = LuongAttnDecoderRNN(
            'dot',
            self.embedding,
            self.hidden_size,
            self.vocab_size,
            n_layers=self.n_layers,
            dropout=0.1
        )

    def test_decoder_initialization(self):

        self.assertEqual(self.decoder.hidden_size, self.hidden_size)
        self.assertEqual(self.decoder.output_size, self.vocab_size)
        self.assertEqual(self.decoder.n_layers, self.n_layers)
        self.assertIsInstance(self.decoder.gru, nn.GRU)
        self.assertIsInstance(self.decoder.attn, Attn)

    def test_decoder_forward_pass_shape(self):

        # Create dummy inputs
        input_step = torch.randint(0, self.vocab_size, (1, self.batch_size))
        last_hidden = torch.randn(self.n_layers, self.batch_size, self.hidden_size)
        encoder_outputs = torch.randn(self.seq_length, self.batch_size, self.hidden_size)

        # Forward pass
        output, new_hidden = self.decoder(input_step, last_hidden, encoder_outputs)

        # Check output shape: (batch_size, vocab_size)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.vocab_size)

        # Check hidden shape is preserved
        self.assertEqual(new_hidden.shape, last_hidden.shape)

    def test_decoder_output_is_probability_distribution(self):

        input_step = torch.randint(0, self.vocab_size, (1, self.batch_size))
        last_hidden = torch.randn(self.n_layers, self.batch_size, self.hidden_size)
        encoder_outputs = torch.randn(self.seq_length, self.batch_size, self.hidden_size)

        output, _ = self.decoder(input_step, last_hidden, encoder_outputs)

        # Check all outputs are non-negative
        self.assertTrue(torch.all(output >= 0), "Output contains negative values")

        # Check outputs sum to approximately 1 (probability distribution)
        output_sum = output.sum(dim=1)
        self.assertTrue(torch.allclose(output_sum, torch.ones_like(output_sum), atol=1e-5))

    def test_decoder_gradient_flow(self):

        self.decoder.train()

        input_step = torch.randint(0, self.vocab_size, (1, self.batch_size))
        last_hidden = torch.randn(self.n_layers, self.batch_size, self.hidden_size)
        encoder_outputs = torch.randn(self.seq_length, self.batch_size, self.hidden_size)

        output, new_hidden = self.decoder(input_step, last_hidden, encoder_outputs)

        # Create dummy loss
        loss = output.sum() + new_hidden.sum()
        loss.backward()

        # Check that parameters have gradients
        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestSeq2SeqIntegration(unittest.TestCase):


    def setUp(self):

        self.hidden_size = 256
        self.vocab_size = 100
        self.n_layers = 2
        self.batch_size = 4
        self.seq_length = 10

        # Create shared embedding
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # Create encoder and decoder
        self.encoder = EncoderRNN(
            self.hidden_size,
            self.embedding,
            n_layers=self.n_layers,
            dropout=0.1
        )
        self.decoder = LuongAttnDecoderRNN(
            'dot',
            self.embedding,
            self.hidden_size,
            self.vocab_size,
            n_layers=self.n_layers,
            dropout=0.1
        )

    def test_encoder_decoder_compatibility(self):

        # Encode
        input_seq = torch.randint(0, self.vocab_size, (self.seq_length, self.batch_size))
        input_lengths = torch.tensor([10, 8, 6, 4])
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)

        # Use encoder hidden as decoder initial hidden
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Decode one step
        decoder_input = torch.randint(0, self.vocab_size, (1, self.batch_size))
        decoder_output, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # Should complete without error
        self.assertEqual(decoder_output.shape[0], self.batch_size)
        self.assertEqual(decoder_output.shape[1], self.vocab_size)

    def test_full_sequence_decoding(self):

        max_decode_length = 15

        # Encode
        input_seq = torch.randint(0, self.vocab_size, (self.seq_length, self.batch_size))
        input_lengths = torch.tensor([10, 8, 6, 4])
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)

        # Decode sequence
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.LongTensor([[SOS_TOKEN]] * self.batch_size).transpose(0, 1)

        all_outputs = []
        for t in range(max_decode_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            all_outputs.append(decoder_output)

            # Get top predicted token
            _, topi = decoder_output.topk(1)
            decoder_input = topi.transpose(0, 1)

        # Check all outputs generated
        self.assertEqual(len(all_outputs), max_decode_length)

        # Check output shapes
        for output in all_outputs:
            self.assertEqual(output.shape, (self.batch_size, self.vocab_size))

    def test_parameter_updates_after_backward(self):

        self.encoder.train()
        self.decoder.train()

        # Store initial parameters
        encoder_params_before = {
            name: param.clone()
            for name, param in self.encoder.named_parameters()
        }
        decoder_params_before = {
            name: param.clone()
            for name, param in self.decoder.named_parameters()
        }

        # Forward pass
        input_seq = torch.randint(0, self.vocab_size, (self.seq_length, self.batch_size))
        input_lengths = torch.tensor([10, 8, 6, 4])
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.randint(0, self.vocab_size, (1, self.batch_size))
        decoder_output, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

        # Compute loss and backpropagate
        target = torch.randint(0, self.vocab_size, (self.batch_size,))
        loss = nn.CrossEntropyLoss()(decoder_output, target)
        loss.backward()

        # Simulate optimizer step
        with torch.no_grad():
            for param in self.encoder.parameters():
                if param.grad is not None:
                    param -= 0.01 * param.grad
            for param in self.decoder.parameters():
                if param.grad is not None:
                    param -= 0.01 * param.grad

        # Check that parameters changed
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.assertFalse(
                    torch.allclose(param, encoder_params_before[name]),
                    f"Encoder parameter {name} did not update"
                )

        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                self.assertFalse(
                    torch.allclose(param, decoder_params_before[name]),
                    f"Decoder parameter {name} did not update"
                )


class TestModelPersistence(unittest.TestCase):


    def setUp(self):

        self.hidden_size = 128
        self.vocab_size = 50
        self.n_layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.n_layers, 0)
        self.decoder = LuongAttnDecoderRNN(
            'dot', self.embedding, self.hidden_size,
            self.vocab_size, self.n_layers, 0
        )

    def test_save_and_load_encoder(self):

        # Save state dict
        state_dict = self.encoder.state_dict()

        # Create new encoder and load state
        new_encoder = EncoderRNN(self.hidden_size, self.embedding, self.n_layers, 0)
        new_encoder.load_state_dict(state_dict)

        # Test forward pass produces same output
        input_seq = torch.randint(0, self.vocab_size, (5, 2))
        input_lengths = torch.tensor([5, 3])

        with torch.no_grad():
            out1, hid1 = self.encoder(input_seq, input_lengths)
            out2, hid2 = new_encoder(input_seq, input_lengths)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))
        self.assertTrue(torch.allclose(hid1, hid2, atol=1e-6))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
