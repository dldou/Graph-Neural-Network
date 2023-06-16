import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np


class CustomizableMNIST(datasets.MNIST):

    def __init__(self, root='./data', train=True, download=True):
        print("Initializing CustomizableMNIST...")
        if train:
            print("Training set")
        else:
            print("Test set")
        super().__init__(root=root, train=train, download=download)
        print("Init done.\n") 

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        # transforms: ToTensor and Normalize
        transform = transforms.Compose([transforms.CenterCrop(28),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                       ])
        image_tens = transform(image)

        return image_tens, target


def split_indices_into_train_val(trainset, val_set_ratio,
                    shuffle, seed=42):
    """
        Split and shuffle (if set to True) the indices of the data
        Return train, valid indices
    """
    val_set_size = int(len(trainset) * val_set_ratio)

    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(trainset))
    else:
        indices = np.arange(0,len(trainset), len(trainset))
    val_indices   = indices[val_set_size:]
    train_indices = indices[:val_set_size]

    return train_indices, val_indices


    


    #def get_data(self, filepath)