import os
from PIL import Image

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


class CIFAR10BiTransform(CIFAR10):
    def __init__(self, transform2, *args, **kwargs):
        super(CIFAR10BiTransform, self).__init__(*args, **kwargs)
        self.transform2 = transform2

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        img2 = None
        if self.transform2 is not None:
            img2 = self.transform2(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if img2 is not None:
            return img, target, img2
        else:
            return img, target


class CIFAR10ResnetDino:
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

        self.transform_dino = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def data_loaders(self, **kwargs):
        trainset = CIFAR10BiTransform(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
            transform2=self.transform_dino
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[: int(self.args.data_fraction * len(trainset))]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = CIFAR10BiTransform(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
            transform2=self.transform_dino
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader


if __name__ == "__main__":

    pretrained_method = "dino"

    train = False

    trainset = datasets.CIFAR10(
        root=os.path.join("../../data/", "CIFAR10"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    testset = datasets.CIFAR10(
        root=os.path.join("../../data/", "CIFAR10"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    idx_file = torch.load(f"./idx/{pretrained_method}/pred_test", map_location="cpu")
    print(idx_file.shape)

    for idx in range(5):
        image_idx = torch.where(idx_file == idx)[0]
        print(f"Idx {idx} image number: {len(image_idx)}")

        image_idx = torch.where(idx_file == idx)[0]

        if train:
            targets_idx = torch.tensor(trainset.targets)[image_idx]
        else:
            targets_idx = torch.tensor(testset.targets)[image_idx]

        stat = []
        for label in range(10):
            stat.append((targets_idx == label).int().sum().item())
        print(f"Idx: {idx} label distribution:")
        print(stat)
