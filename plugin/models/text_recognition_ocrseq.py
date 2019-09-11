import torch.nn as nn
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    BaseFairseqModel,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTM
from fairseq.models.transformer import TransformerDecoder, Embedding

from plugin.models.text_recognition_encoder import ImageEncoder


@register_model('text_recognition_ocrseq')
class TextRecognitionOCRSeq(BaseFairseqModel):
    """
    CRNN model from `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" (Shi, et al, 2015)
    <https://arxiv.org/abs/1507.05717>`_.

    Args:
        encoder (ImageEncoder): the encoder
        decoder (CRNNDecoder): the decoder

    The CRNN model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.text_recognition_crnn_parser
        :prog:
    """

    def __init__(self, cnn_encoder, rnn_encoder, projector_middle, decoder):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.rnn_encoder = rnn_encoder
        self.projector_middle = projector_middle
        self.decoder = decoder
        # assert isinstance(self.encoder, FairseqEncoder)
        # assert isinstance(self.decoder, FairseqDecoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--backbone', default='densenet121',
                            help='CNN backbone architecture. (default: densenet121)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-token-rnn', default=False, action='store_true',
                            help='if set, disables rnn layer')
        parser.add_argument('--no-token-crf', default=False, action='store_true',
                            help='if set, disables conditional random fields')
        parser.add_argument('--decoder-bidirectional', action='store_true',
                            help='make all layers of decoder bidirectional')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 512
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 512

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

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

        # decoder using condition language model
        # decoder = CRNNDecoder(
        #     args=args,
        #     dictionary=task.target_dictionary,
        #     embed_dim=args.decoder_embed_dim,
        #     hidden_size=args.decoder_embed_dim,
        #     bidirectional=args.decoder_bidirectional,
        #     num_layers=args.decoder_layers,
        #     no_token_rnn=args.no_token_rnn,
        # )

        decoder_embed_tokens = build_embedding(task.target_dictionary,
                                               args.decoder_embed_dim)

        decoder = TransformerDecoder(args, task.target_dictionary, decoder_embed_tokens)

        return cls(cnn_encoder, rnn_encoder, projector_middle, decoder)

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        cnn_feature = self.cnn_encoder(src_tokens)
        rnn_feature = self.rnn_encoder(cnn_feature)
        encoder_out = self.projector_middle(rnn_feature)

        decoder_input = {
            'encoder_out': rnn_feature,
            'encoder_padding_mask': None
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out=decoder_input)
        return {
            'encoder_output': encoder_out,
            'decoder_output': decoder_out,
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

    # def get_normalized_probs(self, net_output, log_probs, sample=None):
    #     """Get normalized probabilities (or log probs) from a net's output."""
    #     if hasattr(self, 'decoder'):
    #         return self.decoder.get_normalized_probs(net_output, log_probs, sample)
    #     elif torch.is_tensor(net_output):
    #         logits = net_output.float()
    #         if log_probs:
    #             return F.log_softmax(logits, dim=-1)
    #         else:
    #             return F.softmax(logits, dim=-1)
    #     raise NotImplementedError

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


class CRNNDecoder(FairseqDecoder):
    """CRNN decoder."""

    def __init__(
            self, args, dictionary, embed_dim, hidden_size=512,
            bidirectional=True, num_layers=2, no_token_rnn=False,
    ):
        super().__init__(dictionary)
        self.need_rnn = not no_token_rnn
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=args.dropout)
        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        ) if self.need_rnn else None

        hidden_size = self.hidden_size if self.need_rnn else embed_dim
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, len(dictionary))

    def forward(self, encoder_out, **kwargs):
        # encoder_out -> decoder
        encoder_out = encoder_out['encoder_out']  # seq_len x bsz x embed_dim
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        output, final_hidden = self.rnn(encoder_outs)
        output = self.dropout(self.fc_1(output))
        output = self.classifier(output)
        return output

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        pass

    def output_layer(self, features, **kwargs):
        pass


@register_model_architecture('text_recognition_ocrseq', 'text_recognition_ocrseq')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.backbone = getattr(args, 'backbone', 'densenet121')
    args.pretrained = getattr(args, 'pretrained', True)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.no_token_rnn = getattr(args, 'no_token_rnn', False)
    args.no_token_crf = getattr(args, 'no_token_crf', True)
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', args.dropout)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_input = getattr(args, 'adaptive_input', False)


@register_model_architecture('text_recognition_ocrseq', 'ocrseq')
def decoder_crnn(args):
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', True)
    base_architecture(args)
