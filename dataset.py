import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from scipy.spatial.distance import cdist

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
    
    def get_item_numpy(self, index):
        image, target = super().__getitem__(index)
        # transforms: ToTensor and Normalize
        transform = transforms.Compose([transforms.CenterCrop(28),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                       ])
        image_tens = transform(image)
        image_numpy = image_tens.permute(1,2,0).numpy()

        return image_numpy, target



def compute_adj_mat(image):
    """
        image: numpy array
    """
    col, row = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    coord = np.stack((col, row), axis=2).reshape(-1, 2)
    dist = cdist(coord, coord)
    adj = ( dist <= np.sqrt(2) ).astype(float)
    return adj
    

def norm_adjacency(adj):
    """
        adj: numpy array
    """
    deg = np.diag(np.sum(adj, axis=0))
    deg_inv_1_2 = np.linalg.inv(deg) ** (1/2)
    return deg_inv_1_2 @ adj @ deg_inv_1_2


def split_indices_into_two_sets(trainset, val_set_ratio,
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



def split_and_shuffle_data(set_to_split, split_ratio, batch_size):
    
    biggest_set_indices, littliest_set_indices = split_indices_into_two_sets(set_to_split, split_ratio, shuffle=True, seed=42)
    
    biggest_set_sampler = SubsetRandomSampler(biggest_set_indices)
    littliest_set_sampler = SubsetRandomSampler(littliest_set_indices)

    biggest_set_loader = DataLoader(set_to_split, batch_size=batch_size, sampler=biggest_set_sampler)
    littliest_set_loader = DataLoader(set_to_split, batch_size=batch_size, sampler=littliest_set_sampler)

    return biggest_set_loader, littliest_set_loader

