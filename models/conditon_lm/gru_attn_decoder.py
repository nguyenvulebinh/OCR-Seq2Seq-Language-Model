import torch
import torch.nn as nn
import torch.nn.functional as F
# for unittest
import unittest


class Attn(nn.Module):
    def __init__(self, method, decoder_hidden_size, encoder_hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        # self.hidden_size = hidden_size
        if self.method == 'dot':
            assert decoder_hidden_size == encoder_hidden_size, \
                "Dot attention need decoder_hidden_size == encoder_hidden_size. " \
                "{} != {}".format(decoder_hidden_size, encoder_hidden_size)
        elif self.method == 'general':
            self.attn = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(decoder_hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class GRUAttnDecoder(nn.Module):
    def __init__(self, attn_model, embedding_dim, vocab_size, encoder_hidden_size, decoder_hidden_size,
                 num_layers=1, dropout=0.1):
        super(GRUAttnDecoder, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = decoder_hidden_size
        # self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define layers
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))

        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=decoder_hidden_size,
                          num_layers=num_layers,
                          dropout=(0 if num_layers == 1 else dropout))

        self.concat = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        self.out = nn.Linear(decoder_hidden_size, vocab_size)

        self.attn = Attn(attn_model, decoder_hidden_size, encoder_hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class TestGRUAttnDecoder(unittest.TestCase):

    def test_attn_decoder_forward(self):
        step, batch_size, encoder_hidden_size, num_encoder_layers = 512, 5, 128, 2
        vocab_size = 70
        embedding_dim = 128
        attn_type = 'concat'

        outputs_encode = torch.rand((step, batch_size, encoder_hidden_size))
        hidden_encode = torch.rand((num_encoder_layers, batch_size, encoder_hidden_size * 2))

        decoder_hidden_size = encoder_hidden_size * 2
        num_decoder_layers = num_encoder_layers
        decoder = GRUAttnDecoder(attn_type,
                                 embedding_dim,
                                 vocab_size,
                                 encoder_hidden_size,
                                 decoder_hidden_size,
                                 num_layers=num_decoder_layers)

        input_step_decoder = torch.zeros((1, batch_size), dtype=torch.int64)
        # input_step_decoder_embed = embedding(input_step_decoder)
        output, hidden = decoder(input_step_decoder, hidden_encode, outputs_encode)

        self.assertEqual(tuple(output.shape), (batch_size, vocab_size))
        self.assertEqual(tuple(hidden.shape), (num_decoder_layers, batch_size, decoder_hidden_size))


if __name__ == '__main__':
    unittest.main()
