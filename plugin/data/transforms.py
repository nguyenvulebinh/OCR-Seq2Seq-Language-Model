import numpy as np
from PIL import Image


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


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from plugin.data import data_utils
    import scipy.misc

    transform = transforms.Compose([
        ResizeWithPad(width=1280, height=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = data_utils.default_loader(
        '../../data-bin/ocr-dataset/train/large_data_test_6674.jpg')
    image = transform(image)
    scipy.misc.imsave('outfile.jpg', image.permute(1, 2, 0).numpy())
