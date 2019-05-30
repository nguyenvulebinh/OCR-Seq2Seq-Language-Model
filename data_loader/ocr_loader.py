from .ocr.ocr_dataset import OCRDataset, ResizeWithPad
from .lm.vocabulary import Vocabulary
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json


def get_dataloader(root_data_path, image_width, image_height, vocab, batch_size=4, num_workers=4, shuffle=True):
    """
    create dataset and dataloader for create batch
    :param root_data_path:
    :param image_width:
    :param image_height:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :param vocab:
    :return:
    """
    data_transform = transforms.Compose([
        ResizeWithPad(width=image_width, height=image_height),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ocr_dataset = OCRDataset(root=root_data_path,
                             transform=data_transform,
                             vocab=vocab)

    dataset_loader = DataLoader(ocr_dataset,
                                batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers)

    return dataset_loader


def get_or_create_vocab(root_data_path, labels_file='labels.json', vocab_file=None):
    if vocab_file and os.path.isfile(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as file:
            vocab_data = json.loads(file.read())
        return Vocabulary.from_serializable(vocab_data)
    else:
        with open(os.path.join(root_data_path, labels_file), 'r', encoding='utf-8') as file:
            labels_data = json.loads(file.read())
        vocab = Vocabulary()

        for label in labels_data.values():
            for char in label:
                vocab.add_token(char)

        with open(vocab_file, 'w', encoding='utf-8') as file:
            json.dump(vocab.to_serializable(), file)
        return vocab
