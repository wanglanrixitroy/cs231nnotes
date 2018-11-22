import pandas as pd
import os
from os.path import abspath, dirname, normpath, join
import sys
sys.path.insert(0, normpath(join(abspath(dirname("__file__")))))
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image

def _makelabelset(csv_path):
    landmarks_frame = pd.read_csv(csv_path, sep=',')
    breeds = []
    for name in landmarks_frame.breed:
        if name not in breeds:
            breeds.append(name)
    classes = list(range(len(breeds)))
    labelsset = dict(zip(breeds, classes))
    return labelsset, breeds

class One_hot_label(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.length = 0

    def add_word(self, word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.length + 1
            self.length += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def onehot_encoded(self, word):
        vec = np.zeros(self.length)
        vec[self.word2idx[word]] = 1
        return vec

class DogbreedDataset(Dataset):
    """Read in Dog Breed Dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        local_path = normpath(join(abspath(dirname("__file__"))))
        csv_file = join(local_path, csv_file)
        root_dir = join(local_path, root_dir)
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks_frame = pd.read_csv(self.csv_file, sep=',')
        self.labelset, _ = _makelabelset(csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # define the name of pic
        _img_name = self.landmarks_frame.iloc[idx, 0] + '.jpg'
        img_name = join(self.root_dir, _img_name)

        # read the jpg
        # image = io.imread(img_name)
        image = Image.open(img_name)
        # define the label
        landmarks = self.labelset[self.landmarks_frame.iloc[idx, 1]]
        dataset = {'image': image, 'landmarks': landmarks}

        if self.transform:
            dataset = self.transform(dataset)

        return dataset
