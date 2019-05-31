import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):
    """
    VGG for extract image features. Parameter is frozen
    """

    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)

        # Freeze training for all layers
        for param in vgg16.features.parameters():
            param.requires_grad = False

        self.cnn = vgg16.features

    def forward(self, image):
        features = self.cnn(image)
        return features.view(features.size(0), features.size(1), -1)
