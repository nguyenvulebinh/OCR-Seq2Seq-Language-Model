import math

import torch
import torch.nn as nn

from fairseq.models import FairseqEncoder

from plugin.modules.features_getter import ConvFeaturesGetter


class ImageEncoder(FairseqEncoder):
    """Text Recognition encoder."""

    def __init__(self, args):
        super(FairseqEncoder, self).__init__()
        self.features = ConvFeaturesGetter(args)
        _, self.time_steps, input_size = self.features(torch.rand(1, 3, args.height, args.width)).size()
        self.liner = nn.Linear(input_size, args.decoder_embed_dim)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        feature = self.dropout(self.features(src_tokens))
        feature = self.dropout(torch.relu(self.liner(feature)))
        feature = feature.permute(1, 0, 2).contiguous()  # seq_len x bsz x embed_dim
        final_hiddens, final_cells = None, None

        return {
            'encoder_out': (feature, final_hiddens, final_cells),
            'encoder_padding_mask': None,
        }

    def max_positions(self):
        """Maximum sequence length supported by the encoder."""
        return self.time_steps
