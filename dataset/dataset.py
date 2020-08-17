"""
this file contains class: MiniImagenet and Omniglot,
    each class inherit Dataset, and contains __getitem__ and __len__ method;

"""

from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


class MiniImagenet(Dataset):
    def __init__(self, mode, args):
        """it should contain 
        file_path: image file path
        transform

        Args:
            mode (str, optional): [description]. Defaults to 'train'.
            root (str, optional): [description]. Defaults to ''.
        """
        root_dir = '/test/0Dataset_others/Dataset/mini_imagenet'
        # root_dir = root    # '/test/0Dataset_others/Dataset/mini_imagenet'
        csv_path = os.path.join(root_dir, mode+'.csv')
        # print(csv_path)
        self.data, self.label = get_data_label(root_dir, csv_path)

        # implement transform and image file path
        # self.transform = None
        # Transformation
        if args.model_type == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.model_type == 'ResNet':
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')



    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



def get_data_label(root_dir, csv_path):
    data = []
    label = []
    wnid_list = []
    lb = -1
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    for l in lines:
        name, wnid = l.split(',')
        path = os.path.join(root_dir, 'images', name)
        if wnid not in wnid_list:
            wnid_list.append(wnid)
            lb += 1
        data.append(path)
        label.append(lb)
    return data, label