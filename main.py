import data_loader.ocr.ocr_dataset as ocr_loader
import matplotlib.pyplot as plt
import torch
from models.crnn import CRNN
from data_loader import ocr_loader

# for batch in ocr_loader.get_dataloader('./data-bin/raw/0916_Data Samples 2'):
#     print(batch[0].shape)
#     print(batch[1])
#     break

vocab = ocr_loader.get_or_create_vocab('./data-bin/raw/0916_Data Samples 2', vocab_file='vocab.json')
print(vocab.lookup_token('â'))

# image_width, image_height, image_channel = 1280, 60, 3
# image = torch.rand(4, image_channel, image_width, image_height)
# crnn = CRNN(image_width, image_height, image_channel, 256, 70)
# print(crnn)
# print(crnn(image).shape)
