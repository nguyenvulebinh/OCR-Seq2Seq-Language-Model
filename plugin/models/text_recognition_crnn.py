import torch.nn as nn
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTM

from plugin.models.text_recognition_encoder import ImageEncoder


@register_model('text_recognition_crnn')
class TextRecognitionCRNN(BaseFairseqModel):
    """
    TextRecognitionCRNN extract feature from image and generate to sequence of character

    """

    def __init__(self, cnn_encoder, rnn_encoder, projector_middle):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.rnn_encoder = rnn_encoder
        self.projector_middle = projector_middle

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--backbone', default='densenet121',
                            help='CNN backbone architecture. (default: densenet121)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        parser.add_argument('--remove-tone', action='store_true',
                            help='remove-tone for crnn decoder, no for ocr using crnn')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 512
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 512

        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        # extract image feature using cnn
        cnn_encoder = ImageEncoder(
            args=args,
        )
        # rnn_encoder
        rnn_encoder = RNNEncoder(args,
                                 args.decoder_embed_dim,
                                 args.decoder_embed_dim,
                                 args.decoder_bidirectional,
                                 args.decoder_layers)
        # transformer_encoder

        # projector to predicting in the middle model
        projector_middle = ProjectorMiddle(task.target_dictionary, args.decoder_embed_dim)

        return cls(cnn_encoder, rnn_encoder, projector_middle)

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        cnn_feature = self.cnn_encoder(src_tokens)
        rnn_feature = self.rnn_encoder(cnn_feature)
        encoder_out = self.projector_middle(rnn_feature)

        return {
            'encoder_output': encoder_out,
        }

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (512, 512)

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return 512

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)


class RNNEncoder(nn.Module):
    def __init__(self, args, embed_dim, hidden_size, bidirectional=True, num_layers=2):
        super().__init__()
        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, encoder_out, **kwargs):
        # encoder_out -> decoder
        encoder_out = encoder_out['encoder_out']  # seq_len x bsz x embed_dim
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        output, final_hidden = self.rnn(encoder_outs)
        output = self.dropout(self.linear(output))
        return output


class ProjectorMiddle(nn.Module):
    def __init__(self, dictionary, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, len(dictionary))

    def forward(self, feature, **kwargs):
        output = self.classifier(feature)
        return output

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)


# class CRNNDecoder(FairseqDecoder):
#     """CRNN decoder."""
#
#     def __init__(
#             self, args, dictionary, embed_dim, hidden_size=512,
#             bidirectional=True, num_layers=2, no_token_rnn=False,
#     ):
#         super().__init__(dictionary)
#         self.need_rnn = not no_token_rnn
#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional
#         self.num_layers = num_layers
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.rnn = LSTM(
#             input_size=embed_dim,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             bidirectional=self.bidirectional,
#         ) if self.need_rnn else None
#
#         hidden_size = self.hidden_size if self.need_rnn else embed_dim
#         self.fc_1 = nn.Linear(hidden_size * 2, hidden_size * 2)
#         self.classifier = nn.Linear(hidden_size * 2, len(dictionary))
#
#     def forward(self, encoder_out, **kwargs):
#         # encoder_out -> decoder
#         encoder_out = encoder_out['encoder_out']  # seq_len x bsz x embed_dim
#         encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
#         output, final_hidden = self.rnn(encoder_outs)
#         output = self.dropout(self.fc_1(output))
#         output = self.classifier(output)
#         return output
#
#     def get_normalized_probs(self, net_output, log_probs, sample):
#         """Get normalized probabilities (or log probs) from a net's output."""
#         if log_probs:
#             return utils.log_softmax(net_output, dim=2)
#         else:
#             return utils.softmax(net_output, dim=2)
#
#     def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
#         pass
#
#     def output_layer(self, features, **kwargs):
#         pass


@register_model_architecture('text_recognition_crnn', 'text_recognition_crnn')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.pretrained = getattr(args, 'pretrained', False)
    args.remove_tone = getattr(args, 'remove_tone', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.attention_dropout = getattr(args, 'attention_dropout', args.dropout)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)


@register_model_architecture('text_recognition_crnn', 'crnn')
def decoder_crnn(args):
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', True)
    base_architecture(args)


if __name__ == "__main__":
    F.linear()
