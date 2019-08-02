from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch


def IMDB_Dataset(reviews_dataset, batch_size):
    train_set = []
    val_set = []
    test_set = []
    for review in reviews_dataset:
        if review[-1] == 'train':
            train_set.append(review[:-1])

        elif review[-1] == 'val':
            val_set.append(review[:-1])

        elif review[-1] == 'test':
            test_set.append(review[:-1])

    # Transform to numpy.ndarray
    train_x = np.array([review[:-1] for review in train_set], dtype='int32')
    train_y = np.array([review[-1] for review in train_set], dtype='int32')

    valid_x = np.array([review[:-1] for review in val_set], dtype='int32')
    valid_y = np.array([review[-1] for review in val_set], dtype='int32')

    test_x = np.array([review[:-1] for review in test_set], dtype='int32')
    test_y = np.array([review[-1] for review in test_set], dtype='int32')

    # Create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # Use pytorch's DataLoader
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    # data_iter = iter(train_loader)
    # sample_x, sample_y = data_iter.next()
    # print('Sample input size: ', sample_x.size())
    # print('Sample input: \n', sample_x)
    # print()
    # print('Sample label size: ', sample_y.size())
    # print('Sample label: \n', sample_y)
    # print(train_x.shape)

    return train_loader, valid_loader, test_loader
