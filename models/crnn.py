import torch
import torch.nn as nn
from .img_feature.vgg import VGG


class VisualAttn(nn.Module):
    """
    self-attn between steps
    """

    def __init__(self, time_steps):
        super(VisualAttn, self).__init__()
        self.liner = nn.Linear(time_steps, time_steps)

    def forward(self, inputs):
        """
        Shape should be (batch_size, time_step, features_size)
        :param inputs:
        :return:
        """
        batch_size, time_step, features_size = inputs.size()
        attn_input = inputs.clone()
        attn_input = attn_input.permute(0, 2, 1).contiguous().view(-1, time_step)
        attn = torch.softmax(self.liner(attn_input), dim=1)
        attn = attn.view(batch_size, features_size, time_step)
        attn = attn.mean(dim=1)
        attn_score = attn.unsqueeze(dim=-1).expand(batch_size, time_step, features_size)
        outputs = inputs * attn_score
        return outputs


class CRNN(nn.Module):
    """
    Model using cnn to extract feature and rnn to generate text characters
    """

    def __init__(self, image_width, image_height, image_channel, hidden_size, vocab_size, num_layers=2, drop=0.3,
                 use_vis_attn=True):
        """

        :param image_width:
        :param image_height:
        :param image_channel:
        :param hidden_size:
        :param vocab_size:
        :param num_layers:
        :param drop:
        :param use_vis_attn: use visual attention
        """
        super(CRNN, self).__init__()
        self.vgg = VGG()
        _, time_steps, input_size = self.vgg(torch.rand(1, image_channel, image_width, image_height)).size()
        self.liner = nn.Linear(input_size, 512)
        input_size = 512
        # input_size, hidden_size, num_layers
        if use_vis_attn:
            self.attn = VisualAttn(time_steps=time_steps)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=drop)
        self.use_vis_attn = use_vis_attn
        self.dropout = nn.Dropout(p=drop)
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc_2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, image, softmax=False):
        feature = self.dropout(self.vgg(image))
        feature = self.dropout(torch.relu(self.liner(feature)))
        if self.use_vis_attn:
            feature = self.attn(feature)
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
