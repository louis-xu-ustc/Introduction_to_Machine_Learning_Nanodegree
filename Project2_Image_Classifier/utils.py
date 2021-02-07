import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_dir):
    """
    load the data from torchvision datasets
    :param data_dir: the directory where data is stored
    :return: train_datasets, valid_datasets, test_datasets, train_loader, valid_loader, test_loader
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(255),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)

    return train_datasets, valid_datasets, test_datasets, train_loader, valid_loader, test_loader


def helper_imshow(image, ax=None, title=None, normalize=True):
    """
    a helper function to show image for Tensor.
    :param image: a image in Tensor format
    :param ax: axis for plots
    :param title: title of the image
    :param normalize: whether to normalize the image
    :return: ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
