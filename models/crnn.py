import torch
import torch.nn as nn
from .img_feature.vgg import VGG


class CRNN(nn.Module):
    """
    Model using cnn to extract feature and rnn to generate text characters
    """

    def __init__(self, image_width, image_height, image_channel, hidden_size, vocab_size, drop=0.3):
        super(CRNN, self).__init__()
        self.vgg = VGG()
        input_size = self.vgg(torch.rand(1, image_channel, image_width, image_height)).size(-1)
        # input_size, hidden_size, num_layers
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=2,
                           bidirectional=True,
                           batch_first=True,
                           dropout=drop)
        self.dropout = nn.Dropout(p=drop)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, image, apply_softmax=False):
        feature = self.dropout(self.vgg(image))
        output, final_hidden = self.rnn(feature)
        batch_size, seq_length, feature_size = output.size()
        output = output.contiguous().view(batch_size * seq_length, feature_size)
        output = self.fc(output)
        if apply_softmax:
            output = torch.softmax(output, dim=1)

        output = output.view(batch_size, seq_length, -1)
        return output