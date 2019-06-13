import torch
import torch.nn as nn
from .img_feature.vgg import VGG
from models.img_feature.gru_encoder import GRUEncoder
from models.conditon_lm.gru_attn_decoder import GRUAttnDecoder
# for unittest
import unittest
from data_loader.lm.vocabulary import Vocabulary


class GRUEncodeDecode(nn.Module):
    """
    Model using cnn to extract feature and rnn to generate text characters
    """

    def __init__(self, image_width, image_height, image_channel, encoder_hidden_size, decoder_hidden_size,
                 embedding_dim, vocab_size, max_target_len, vocab, num_layers=2, drop=0.3, device='cpu'):
        super(GRUEncodeDecode, self).__init__()
        self.max_target_len = max_target_len
        self.vgg = VGG()
        self.device = device
        self.vocab = vocab
        input_size = self.vgg(torch.rand(1, image_channel, image_width, image_height)).size(-1)
        self.encoder = GRUEncoder(feature_size=input_size, hidden_size=encoder_hidden_size, num_layers=num_layers,
                                  dropout=drop)
        self.decoder = GRUAttnDecoder(attn_model='general', embedding_dim=embedding_dim, vocab_size=vocab_size,
                                      encoder_hidden_size=encoder_hidden_size, decoder_hidden_size=decoder_hidden_size,
                                      num_layers=num_layers, dropout=drop)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, image, target_teach_force=None):
        batch_size = image.size(0)
        # get image features
        image_features = self.dropout(self.vgg(image))
        # Forward pass through encoder
        encoder_outputs, encoder_hidden_state = self.encoder(image_features)
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden_state
        decoder_outputs = []
        if target_teach_force is not None:
            # using teach force
            # (batch, indices) -> (indices, batch) for loop
            input_indices = target_teach_force.transpose(1, 0)
            for index in range(self.max_target_len):
                input_step = input_indices[index].view(1, -1)
                output_step, decoder_hidden = self.decoder(input_step, decoder_hidden, encoder_outputs)
                decoder_outputs.append(output_step)
        else:
            # normal infer
            input_step = (torch.ones((1, batch_size)) * self.vocab.get_sos_id()).long().to(self.device)
            for index in range(self.max_target_len):
                output_step, decoder_hidden = self.decoder(input_step, decoder_hidden, encoder_outputs)
                decoder_outputs.append(output_step)

                # No teacher forcing: next input is decoder's own current output
                _, topi = output_step.topk(1)
                input_step = torch.tensor([[topi[i][0] for i in range(batch_size)]], dtype=torch.int64).to(self.device)
        decoder_outputs = torch.stack(decoder_outputs, dim=0).transpose(1, 0).contiguous()
        return decoder_outputs


class TestGRUEncodeDecode(unittest.TestCase):

    def test_forward(self):
        image_width = 1280
        image_height = 60
        image_channel = 3
        encoder_hidden_size = 128
        decoder_hidden_size = 256
        embedding_dim = 64
        vocab_size = 20
        max_target_len = 30
        vocab = Vocabulary()
        batch_size = 5
        model = GRUEncodeDecode(image_width, image_height, image_channel, encoder_hidden_size, decoder_hidden_size,
                                embedding_dim, vocab_size, max_target_len, vocab)

        images = torch.rand(batch_size, image_channel, image_width, image_height)
        target_teach_force = torch.randint(low=0, high=vocab_size, size=(batch_size, max_target_len), dtype=torch.int64)
        output_with_teach_force = model(images, target_teach_force)
        output_without_teach_force = model(images)
        self.assertEqual(tuple(output_with_teach_force.shape), (batch_size, max_target_len, vocab_size))
        self.assertEqual(tuple(output_without_teach_force.shape), (batch_size, max_target_len, vocab_size))


if __name__ == '__main__':
    unittest.main()
