import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from models.img_feature.vgg import VGG
import unittest


class GRUEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, dropout=0.3, bidirectional=True):
        """
        RNN encode feature from cnn
        :param feature_size:
        :param hidden_size:
        :param num_layers:
        :param dropout:
        :param bidirectional:
        """
        super(GRUEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.encoder = nn.GRU(input_size=feature_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional)

    def forward(self, image_features):
        """
        :param image_features: (batch, step, feature_size)
        :return: (batch, step, hidden_size*2) ((batch, num_layers, hidden_size*2), (batch, num_layers, hidden_size*2))
        """
        batch, step, feature_size = image_features.size()
        image_features = image_features.transpose(1, 0)
        outputs, hidden = self.encoder(image_features)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Combine forward and backward hidden state
        hidden = hidden.view(self.num_layers, self.num_directions, batch, self.hidden_size)
        hidden = hidden.transpose(1, 2).contiguous().view(self.num_layers, batch, -1)

        return outputs, hidden


class TestGRUEncoder(unittest.TestCase):

    def test_encode_forward(self):
        vgg = VGG()
        image_channel, image_width, image_height = 3, 1280, 60
        batch_size = 5
        gru_hidden_size = 256
        num_gru_layers = 4
        cnn_feature = vgg(torch.rand(batch_size, image_channel, image_width, image_height))
        batch, step, feature_size = cnn_feature.size()
        outputs_encode, hidden_encode = GRUEncoder(feature_size,
                                                   hidden_size=gru_hidden_size,
                                                   num_layers=num_gru_layers)(cnn_feature)
        self.assertEqual(tuple(outputs_encode.shape), (step, batch_size, gru_hidden_size))
        self.assertEqual(tuple(hidden_encode.shape), (num_gru_layers, batch_size, gru_hidden_size * 2))


if __name__ == '__main__':
    unittest.main()
