import torch.nn as nn
import plugin.modules as modules


class ConvFeaturesGetter(nn.Module):
    """CNN features getter for the Encoder of image."""

    def __init__(self, backbone_name, pretrained):
        super().__init__()
        # loading network
        conv_model_in = getattr(modules, backbone_name)(pretrained=pretrained)

        if backbone_name.startswith('vgg'):
            # Freeze training for all layers
            for param in conv_model_in.features[:20].parameters():
                param.requires_grad = False
            conv_model_in = conv_model_in.features[:-1]
            self.conv = conv_model_in
        else:

            if backbone_name.startswith('resnet') or backbone_name.startswith('mobilenet'):
                conv = list(conv_model_in.children())[:-2]
            elif backbone_name.startswith('densenet'):
                conv = list(conv_model_in.features.children())
                conv.append(nn.ReLU(inplace=True))
            else:
                raise ValueError('Unsupported or unknown architecture: {}!'.format(backbone_name))

            self.conv = nn.Sequential(*conv)

    def forward(self, x):
        # return shape (batch_size, feature_1, feature_2, step)
        features = self.conv(x)
        return features.view(features.size(0), features.size(1), -1)
        # return self.conv(x).permute(0, 3, 2, 1).contiguous()
