import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import pandas as pd
import glob
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image


class P2_Train_Dataset(Dataset):
    def __init__(self,dir_path,transform=None):
        super().__init__()
        self.dir_path = dir_path
        self.transforms = transform
        self.df = pd.read_csv("./hw2_data/digits/mnistm/train.csv")


    def __len__(self):
        return self.df.shape[0] #rows
    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.df.iloc[index,0])
        label = self.df.iloc[index,1]
        img = Image.open(img_path).convert("RGB")
        if self.transforms != None:
            img = self.transforms(img)
        return (img,label)


class P3_Train_Dataset(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        super().__init__()
        self.dir_path = data_root
        self.transforms = transform
        self.df = pd.read_csv(data_list)


    def __len__(self):
        return self.df.shape[0] #rows
    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.df.iloc[index,0])
        label = self.df.iloc[index,1]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return (img,label)

class P3_Test_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.img_files = glob.glob(os.path.join(data_dir,"*.png"))
        self.transforms = transform
    

    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        img_path = self.img_files[index]
        label = 0
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return (img,label,img_path.split("/")[-1])

def get_mnistm(dataset_root, batch_size, mode):
    """Get MNISTM datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                     )])

    # datasets and data_loader
    if mode == "train":
        train_list = os.path.join(dataset_root, "train.csv")#../hw2_data/digits/mnistm/train.csv
        mnistm_dataset = P3_Train_Dataset(
            data_root=os.path.join(dataset_root,"data"),#../hw2_data/digits/mnistm/data/
            data_list=train_list,
            transform=pre_process)
    elif mode == "val":
        train_list = os.path.join(dataset_root, "val.csv")#../hw2_data/digits/mnistm/val.csv
        mnistm_dataset = P3_Train_Dataset(
            data_root=os.path.join(dataset_root,"data"),#../hw2_data/digits/mnistm/data/
            data_list=train_list,
            transform=pre_process)

    mnistm_dataloader = torch.utils.data.DataLoader(
        dataset=mnistm_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return mnistm_dataloader

def get_svhn(dataset_root, batch_size, mode):
    """Get SVHN datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])

    # datasets and data_loader
    if mode == "train":
        train_list = os.path.join(dataset_root, "train.csv")#../hw2_data/digits/svhn/train.csv
        svhn_dataset = P3_Train_Dataset(
            data_root=os.path.join(dataset_root,"data"),#../hw2_data/digits/svhn/data/
            data_list=train_list,
            transform=pre_process)
    elif mode == "val":
        train_list = os.path.join(dataset_root, "val.csv")#../hw2_data/digits/mnistm/val.csv
        svhn_dataset = P3_Train_Dataset(
            data_root=os.path.join(dataset_root,"data"),#../hw2_data/digits/svhn/data/
            data_list=train_list,
            transform=pre_process)

    svhn_dataloader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return svhn_dataloader
def get_svhn_test(data_dir, batch_size):
    """Get SVHN datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])
    # datasets and data_loader
    svhn_dataset = P3_Test_Dataset(
        data_dir=data_dir,
        transform=pre_process)
    svhn_dataloader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return svhn_dataloader
def get_usps(dataset_root, batch_size, mode):
    """Get MNISTM datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                     )])

    # datasets and data_loader
    if mode == "train":
        train_list = os.path.join(dataset_root, "train.csv")#../hw2_data/digits/mnistm/train.csv
        usps_dataset = P3_Train_Dataset(
            data_root=os.path.join(dataset_root,"data"),#../hw2_data/digits/mnistm/data/
            data_list=train_list,
            transform=pre_process)
    elif mode == "val":
        train_list = os.path.join(dataset_root, "val.csv")#../hw2_data/digits/mnistm/val.csv
        usps_dataset = P3_Train_Dataset(
            data_root=os.path.join(dataset_root,"data"),#../hw2_data/digits/mnistm/data/
            data_list=train_list,
            transform=pre_process)
    usps_dataloader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return usps_dataloader

def get_usps_test(data_dir, batch_size):
    """Get MNISTM datasets loader."""
     # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                     )])

    # datasets and data_loader
    usps_dataset = P3_Test_Dataset(
        data_dir=data_dir,
        transform=pre_process)
    usps_dataloader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    return usps_dataloader