import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


class CIFAR10Idx:
    """
            CIFAR-10 dataset.
        """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        train_idx_file = torch.load(f"./data/idx/{self.args.dataset_idx_method}/pred_train", map_location="cpu")
        idx = self.args.dataset_idx
        train_image_idx = torch.where(train_idx_file == idx)[0]
        assert len(train_image_idx) > 0

        print(f"Loading indexed CIFAR-10 with index {idx}.")

        trainset.data = trainset.data[train_image_idx]
        trainset.targets = np.array(trainset.targets)[train_image_idx].tolist()

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
                         : int(self.args.data_fraction * len(trainset))]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )

        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )

        test_idx_file = torch.load(f"./data/idx/{self.args.dataset_idx_method}/pred_test", map_location="cpu")
        test_image_idx = torch.where(test_idx_file == idx)[0]
        assert len(test_image_idx) > 0

        testset.data = testset.data[test_image_idx]
        testset.targets = np.array(testset.targets)[test_image_idx].tolist()

        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class CIFAR10:
    """ 
        CIFAR-10 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
                         : int(self.args.data_fraction * len(trainset))
                         ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


class CIFAR100:
    """ 
        CIFAR-100 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
                         : int(self.args.data_fraction * len(trainset))
                         ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


if __name__ == "__main__":

    tr_train = [
        transforms.ToTensor(),
    ]

    trainset = datasets.CIFAR10(
        root=os.path.join("../../data/", "CIFAR10"),
        train=True,
        download=True,
        transform=tr_train,
    )

    idx_file = torch.load("./idx/wrn/pred_train", map_location="cpu")
    print(idx_file.shape)

    for idx in range(5):
        image_idx = torch.where(idx_file == idx)[0]
        print(len(image_idx))
    idx = 3

    image_idx = torch.where(idx_file == idx)[0]

    trainset.data = trainset.data[image_idx]
    trainset.targets = np.array(trainset.targets)[image_idx].tolist()

    print(trainset.targets)
    print(len(trainset.data))
