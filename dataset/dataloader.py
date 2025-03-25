import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import cv2
import os
import random


# color = [
#     (0, 0, 0),
#     (31, 120, 180)
# ]


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225] 

def normolize_image(image):
    image = image.astype(np.float32)
    image = image / 255.0
    image -= mean
    image /= std
    return image


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  #


class UTE_Dataset(Dataset):
    def __init__(self, data_root, list_file, image_size = (640, 480), transform=None):
        super().__init__()

        self.data_root = data_root
        self.image_label_list = self.make_dataset(list_file)

        color_list = [
            (0, 0, 0),  # backgroud
            (31, 120, 180), # road
            (227, 26, 28),  # person 
            (106, 61, 154), # car
            ]
        self.color_map = {}
        for idx, color_value in enumerate(color_list):
            self.color_map[idx] = color_value

        self.image_size = image_size  #(width, height)
        self.transform = transform


    def __len__(self):
        return len(self.image_label_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_root, self.image_label_list[index][0])
        label_path = os.path.join(self.data_root, self.image_label_list[index][1])


        image = self.cv2_loader(image_path)

        label = self.cv2_loader(label_path)
        label = self.convert_label(label)
        
        # resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)

        if random.random() < 0.5:
            augment_hsv(image)
        # augmentation
        if self.transform:
            transform = self.transform(image=image, mask=label)
            image, label = transform['image'], transform['mask']

        # gen edge
        edge = self.gen_edge(label)

        # to tensor
        image = normolize_image(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        label = torch.from_numpy(label)[:, :, 0].long()
        edge = torch.from_numpy(edge).float()
        
        return image, label, edge
    
    def convert_label(self, label):
        temp = label.copy()
        label_ = np.zeros_like(label)
        for idx, color_value in self.color_map.items():
            label_[(temp == color_value)] = idx
        
        return label_
    
    def gen_edge(self, label):
        edge_size = 4
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        return edge
    
    def cv2_loader(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def make_dataset(self, list_file):

        all_list = []
        for file in list_file:
            datas = open(file).readlines()
            image_label_list = [ (data.split()[0], data.split()[1]) for data in datas]

            all_list += image_label_list

        return all_list
