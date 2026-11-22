"""Six-class road surface dataset (RSCD reorganised), stored as ImageFolder:

    <root>/{train,val,test}/{dry,wet,water,fresh_snow,melted_snow,ice}/*.jpg
"""
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ("dry", "fresh_snow", "ice", "melted_snow", "water", "wet")  # ImageFolder order

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def eval_transform(size: int = 224):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), _NORMALIZE])


def train_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def _folder(data_dir: str, split: str, transform) -> datasets.ImageFolder:
    path = os.path.join(os.path.abspath(data_dir), split)
    if not os.path.isdir(path):
        raise FileNotFoundError("Split not found: {}".format(path))
    dataset = datasets.ImageFolder(path, transform=transform)
    if tuple(dataset.classes) != CLASSES:
        raise ValueError("Unexpected classes {} in {}".format(dataset.classes, path))
    return dataset


def get_train_valid_loader(data_dir, batch_size, augment=True, shuffle=True,
                           num_workers=4, pin_memory=True):
    train_set = _folder(data_dir, "train", train_transform() if augment else eval_transform())
    val_set = _folder(data_dir, "val", eval_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=True):
    return DataLoader(_folder(data_dir, "test", eval_transform()), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
