from skimage import transform
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

class Rescale(object):
    """Rescale the image in a sample to a given size

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of iamge edges is matched to output_size keeping ratio the same
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img = np.asarray(image)
        h, w = img.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w))

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': iamge, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        trans = transforms.ToTensor()
        img = trans(image)
        num_classes = 120
        batch_size = 1
        label = torch.LongTensor([[landmarks]])
        # one hot
        landmarks = torch.zeros(batch_size, num_classes).scatter_(1, label, 1)
        return {'image': img, 'landmarks': landmarks}

class Normalize(object):
    """Normalize a tesnsor image with mean and standard deviation.
    Given mean: ``(M1, ..., Mn)`` and std: ``(S1, ..., Sn)`` for ``n`` channels, this transform will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, landmarks = sample['image'], sample['landmarks']
        # for BECLoss function
        landmarks = landmarks.type(torch.FloatTensor).squeeze()

        # landmarks = landmarks.type(torch.LongTensor)

        return {'image':F.normalize(image, self.mean, self.std), 'landmarks': landmarks}
