import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def splitdata(dataset, train_ratio, validate_ratio, random_seed):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = int(validate_ratio * dataset_size)
    test_size = int(dataset_size - train_size - validation_size)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + validation_size)]
    test_indices = indices[(train_size + validation_size):]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, valid_sampler, test_sampler

# class SplitData(object):
#     """Split the dataset into train, validation and test"""
#
#     def __call__(dataset):
