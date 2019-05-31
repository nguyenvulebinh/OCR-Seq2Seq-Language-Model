import data_loader.ocr.ocr_dataset as ocr_loader
import matplotlib.pyplot as plt
import torch
from models.crnn import CRNN
from data_loader import ocr_loader
from trainer import crnn_trainer
import json
import torch.nn.functional as F
from trainer.crnn_trainer import CRNNTrainer

# vocab = ocr_loader.get_or_create_vocab('./data-bin/raw/0916_Data Samples 2', vocab_file='vocab.json')
# for batch in ocr_loader.get_dataloader('./data-bin/raw/0916_Data Samples 2', vocab):
#     print(batch)
#     print(batch.get('labels_lengths').shape)
#     # print(batch[1])
#     # for label_indices in batch[1]:
#     #     print(vocab.ids_to_sentence(label_indices.numpy()))
#     break


# image_width, image_height, image_channel = 1280, 60, 3
# image = torch.rand(4, image_channel, image_width, image_height)
# crnn = CRNN(image_width, image_height, image_channel, 256, 70)
# print(crnn)
# print(crnn(image).shape)

# with open('./config/data.json', encoding='utf-8') as file_config:
#     data_config = json.load(file_config.read())
#
# with open('./config/model_crnn.json', encoding='utf-8') as file_config:
#     model_config = json.load(file_config.read())
#
# with open('./config/trainer.json', encoding='utf-8') as file_config:
#     trainer_config = json.load(file_config.read())
#
# crnn_trainer.CRNNTrainer()


# output = torch.tensor([[[0.0, 10000, 0.0, 0.0],
#                         [0.0, 0.0, 10000, 0.0],
#                         [10000, 0.0, 0.0, 0.0]],
#
#                        [[0.0, 10000, 0.0, 0.0],
#                         [0.0, 0.0, 10000, 0.0],
#                         [0.0, 0.0, 10000, 0.0]]]).log_softmax(2)
# print(output)
# # 012
# # 020
#
# labels = torch.tensor([[1, 2, 3],
#                        [1, 2, 3]])
# input_lengths = torch.tensor([3, 3], dtype=torch.long)
# output_lengths = torch.tensor([2, 2], dtype=torch.long)
# print(labels)
# # print(output_lengths)
# batch_size, steps_lengths, vocab_size = output.size()
# print(output.transpose(0, 1))
# loss = F.ctc_loss(output.transpose(0, 1),
#                   labels,
#                   input_lengths,
#                   output_lengths,
#                   blank=0)
#
# print(loss)

# log_probs = torch.randn(50, 16, 20).log_softmax(2)
# targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
# input_lengths = torch.full((16,), 50, dtype=torch.long)
# target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
# loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
# print(loss)
# print(torch.randint(1, 20, (16, 30), dtype=torch.long))

def read_file_config(file_path):
    with open(file_path, encoding='utf-8') as file_config:
        content = file_config.read()
    return json.loads(content)


data_config = read_file_config('./config/data.json')
model_config = read_file_config('./config/model_crnn.json')
trainer_config = read_file_config('./config/trainer.json')

trainer = CRNNTrainer(trainer_config, model_config, data_config)
trainer.train()
