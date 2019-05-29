import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):
    """
    VGG for extract image features. Parameter is frozen
    """

    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = models.vgg16_bn()
        # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))

        # Freeze training for all layers
        for param in vgg16.features.parameters():
            param.require_grad = False

        self.cnn = vgg16.features

    def forward(self, image):
        features = self.cnn(image)
        return features.view(features.size(0), features.size(1), -1)
