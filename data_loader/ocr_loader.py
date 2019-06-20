from .ocr.ocr_dataset import OCRDataset, ResizeWithPad
from .lm.vocabulary import Vocabulary
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
from utils import text_utils
import glob
from PIL import Image
import torch


def get_dataloader(root_data_path, image_width, image_height, vocab, batch_size=4, num_workers=4, keep_tone=True,
                   shuffle=True, max_target_len=None):
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
                             vocab=vocab,
                             keep_tone=keep_tone,
                             max_length=max_target_len)

    dataset_loader = DataLoader(ocr_dataset,
                                batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers)

    return dataset_loader


def get_infer_data(image_width, image_height, data_path=None, image_list=None, batch_size=4):
    """
    convert list images path or folder images to batch for infer model
    :param image_width:
    :param image_height:
    :param data_path:
    :param image_list:
    :param batch_size:
    :return:
    """

    data_transform = transforms.Compose([
        ResizeWithPad(width=image_width, height=image_height),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not image_list and not data_path:
        raise Exception('require data_path or image_list not none')

    if not image_list:
        image_list = []
        image_list.extend(glob.glob(os.path.join(data_path, "*.jpg")))
        image_list.extend(glob.glob(os.path.join(data_path, "*.png")))

    for i in range(0, len(image_list), batch_size):
        batch_imgs_name = image_list[i:i + batch_size]
        images_matrix = []
        for path in batch_imgs_name:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = data_transform(img.convert('RGB'))
                images_matrix.append(img)

        yield batch_imgs_name, torch.stack(images_matrix)


def get_or_create_vocab(root_data_path=None, labels_file='labels.json', keep_tone=True, vocab_file=None):
    if vocab_file and os.path.isfile(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as file:
            vocab_data = json.loads(file.read())
        return Vocabulary.from_serializable(vocab_data)
    else:
        with open(os.path.join(root_data_path, labels_file), 'r', encoding='utf-8') as file:
            labels_data = json.loads(file.read())
        vocab = Vocabulary()

        if keep_tone:
            for label in labels_data.values():
                for char in label:
                    vocab.add_token(char)
        else:
            for label in labels_data.values():
                for char in text_utils.remove_tone_line(label):
                    vocab.add_token(char)

        with open(vocab_file, 'w', encoding='utf-8') as file:
            json.dump(vocab.to_serializable(), file)
        return vocab
