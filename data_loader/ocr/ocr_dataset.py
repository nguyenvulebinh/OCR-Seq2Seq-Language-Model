from torchvision import datasets
import numpy as np
from PIL import Image
import json
import torch
import os
from utils import text_utils


class ResizeWithPad(object):
    """
    Resize image but keep dim and add pad background
    """

    def __init__(self, width, height):
        """
        Args
        :param width: image width
        :param height: image height
        :return:
        """
        self.width = width
        self.height = height

    def change_dim(self, image):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        (w, h) = image.size
        # if h > height and w > width:
        rate_width = w / self.width
        rate_height = h / self.height
        rate = rate_width
        if h / rate_width > self.height:
            rate = rate_height
        dim = (int(w / rate), int(h / rate))
        # resize the image
        resized = image.resize(dim)
        # return the resized image
        return resized

    def __call__(self, image):
        image = self.change_dim(image)
        # return image
        (w_i, h_i) = image.size
        array_background = np.ndarray((self.height, self.width, 3), dtype=np.uint8)
        np.ndarray.fill(array_background, 0)
        background_img = Image.fromarray(array_background)
        background_img.paste(image, box=(self.width // 2 - w_i // 2, self.height // 2 - h_i // 2), mask=None)
        return background_img


class OCRDataset(datasets.VisionDataset):
    """
    Load images and labels from folder
    folder sample:
    ---root
        |---img_1.png
        |---img_2.png
        |---img_3.png
        |---img_4.png
        |---...
        |---labels.json
    """

    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
                 labels_file='labels.json', vocab=None, max_length=None, keep_tone=True):
        super(OCRDataset, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = datasets.folder.IMG_EXTENSIONS
        self.vocab = vocab
        samples = self.make_dataset(self.root, self.extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                self.extensions)))

        self.loader = datasets.folder.default_loader
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.max_label_length = max_length
        self.keep_tone = keep_tone
        try:
            with open(os.path.join(root, labels_file), 'r', encoding='utf-8') as file:
                self.labels = json.loads(file.read())
            if max_length is None:
                self.max_label_length = 0
                for label in self.labels.values():
                    if len(label) > self.max_label_length:
                        self.max_label_length = len(label)
        except:
            self.labels = None

    @staticmethod
    def make_dataset(dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return datasets.folder.has_file_allowed_extension(x, extensions)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    images.append(path)
        return images

    def __len__(self):
        # return 10
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is label of the sample
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        file_name = path[path.rindex(os.path.sep) + 1:]

        if self.keep_tone:
            target = self.labels.get(file_name)
        else:
            target = text_utils.remove_tone_line(self.labels.get(file_name))

        target_vector = self.vocab.sentence_to_ids(target, self.max_label_length)

        return {
            "images": sample,
            "labels": target_vector,
            "labels_lengths": self.max_label_length
            # "labels_lengths": len(target)
        }
