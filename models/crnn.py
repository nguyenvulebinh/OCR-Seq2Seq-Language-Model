import torch
import torch.nn as nn
from .img_feature.vgg import VGG


class CRNN(nn.Module):
    """
    Model using cnn to extract feature and rnn to generate text characters
    """

    def __init__(self, image_width, image_height, image_channel, hidden_size, vocab_size, num_layers=2, drop=0.3):
        super(CRNN, self).__init__()
        self.vgg = VGG()
        input_size = self.vgg(torch.rand(1, image_channel, image_width, image_height)).size(-1)
        # input_size, hidden_size, num_layers
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=drop)
        self.dropout = nn.Dropout(p=drop)
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc_2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, image, softmax=False):
        feature = self.dropout(self.vgg(image))
        output, final_hidden = self.rnn(feature)
        batch_size, seq_length, feature_size = output.size()
        output = output.contiguous().view(batch_size * seq_length, feature_size)
        output = self.dropout(self.fc_1(output))
        output = self.fc_2(output)
        if softmax:
            output = output.view(batch_size, seq_length, -1).softmax(2)
        else:
            output = output.view(batch_size, seq_length, -1).log_softmax(2)
        return output
