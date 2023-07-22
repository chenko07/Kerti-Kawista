#--- Training ResNet 15 --
# Created by Marchel Shevchenko 
# Date : 21 July 2023

#Library 
import random
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.transforms as T
# import tqdm
import time 
import torch.nn as nn
from torchvision.utils import make_grid
# import re
import os
import requests
import matplotlib.pyplot as plt
import warnings
warnings.warn("ignore")
from torchvision import transforms

class MyRegex:
    def search(self, pattern, text):
        # Implementasi pencarian pola dalam sebuah teks
        for i in range(len(text)):
            if text[i:i + len(pattern)] == pattern:
                return text[i:i + len(pattern)]
        return None

    def match(self, pattern, text):
        # Implementasi pencocokan pola hanya di awal teks
        if text.startswith(pattern):
            return pattern
        return None

    def findall(self, pattern, text):
        # Implementasi pencarian semua kemunculan pola dalam teks
        matches = []
        i = 0
        while i < len(text):
            match = self.match(pattern, text[i:])
            if match:
                matches.append(match)
                i += len(match)
            else:
                i += 1
        return matches

    def sub(self, pattern, replacement, text):
        # Implementasi penggantian pola dengan teks lain
        result = ""
        while True:
            match = self.search(pattern, text)
            if match:
                result += text[:text.index(match)] + replacement
                text = text[text.index(match) + len(match):]
            else:
                result += text
                break
        return result
    
#TQDM Time bar 
class MyTqdm:
    def __init__(self, iterable, total=None, desc='', bar_length=40):
        self.iterable = iterable
        self.total = len(iterable) if total is None else total
        self.desc = desc
        self.bar_length = bar_length
        self.start_time = time.time()

    def __iter__(self):
        return self._generate()

    def _format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02}:{minutes:02}:{seconds:02}'

    def _progress_bar(self, iteration):
        progress = int(self.bar_length * iteration / self.total)
        return f"[{'=' * progress}{' ' * (self.bar_length - progress)}]"

    def _generate(self):
        print(f"{self.desc}: 0% {self._progress_bar(0)} 00:00:00", end='', flush=True)

        for i, item in enumerate(self.iterable, 1):
            yield item
            if i % (self.total // self.bar_length) == 0 or i == self.total:
                elapsed_time = time.time() - self.start_time
                percentage = i / self.total * 100
                remaining_time = (elapsed_time / i) * (self.total - i)
                print(f"\r{self.desc}: {percentage:.1f}% {self._progress_bar(i)} {self._format_time(elapsed_time)} / ETA {self._format_time(remaining_time)}", end='', flush=True)

        print("\n")
        
#Creating OS for Operating System Directory File
class MyOS:
    def __init__(self):
        self.current_directory = os.getcwd()

    def getcwd(self):
        return self.current_directory

    def chdir(self, path):
        if os.path.exists(path) and os.path.isdir(path):
            self.current_directory = path
        else:
            raise FileNotFoundError("Direktori tidak ditemukan.")

    def listdir(self, path='.'):
        full_path = os.path.join(self.current_directory, path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            return os.listdir(full_path)
        else:
            raise FileNotFoundError("Direktori tidak ditemukan.")

    def mkdir(self, path):
        full_path = os.path.join(self.current_directory, path)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
        else:
            raise FileExistsError("Direktori sudah ada.")

    def rename(self, old, new):
        old_path = os.path.join(self.current_directory, old)
        new_path = os.path.join(self.current_directory, new)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        else:
            raise FileNotFoundError("Path lama tidak ditemukan.")

    def remove(self, path):
        full_path = os.path.join(self.current_directory, path)
        if os.path.exists(full_path):
            os.remove(full_path)
        else:
            raise FileNotFoundError("Path tidak ditemukan.")

    def rmdir(self, path):
        full_path = os.path.join(self.current_directory, path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            os.rmdir(full_path)
        else:
            raise FileNotFoundError("Direktori tidak ditemukan atau bukan direktori kosong.")

    def exists(self, path):
        full_path = os.path.join(self.current_directory, path)
        return os.path.exists(full_path)

    def isfile(self, path):
        full_path = os.path.join(self.current_directory, path)
        return os.path.isfile(full_path)

    def isdir(self, path):
        full_path = os.path.join(self.current_directory, path)
        return os.path.isdir(full_path)
    
#Calling Dataset
data_directory = 'Face Recognition'
my_os = MyOS()

try:
    file_list = my_os.listdir(data_directory)
    print(file_list)
except FileNotFoundError as e:
    print(e)

def get_path_names(dir):
    images=[]
    for path, subdir, files in os.walk(data_directory):
        for name in files:
            images.append(os.path.join(path, name))
    return images

path = get_path_names(data_directory)

def get_path_names(root_dir):
    images=[]
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                images.append((image_path, label))
    return images

classes = ['Alvian','Chen','Nicho','Rendie']


def encode_label(label, classes_list):
    if not isinstance(label, list):
        label = [label]
    target = torch.zeros(len(classes_list))
    for l in label:
        if l in classes_list:
            idx = classes_list.index(l)
            target[idx] = 1
    return target


def decode_target(target, threshold=0.5):
    if isinstance(target, list):
        result = ""
        for x in target:
            try:
                value = float(x)
                if value >= threshold:
                    result += "1"
                else:
                    result += "0"
            except ValueError:
                result += x
        return result
    else:
        try:
            value = float(target)
            if value >= threshold:
                return "1"
            else:
                return "0"
        except ValueError:
            return target               
        
        
def get_classes(root_dir):
    classes = []
    for subdir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, subdir)):
            classes.append(subdir)
    return classes

class myDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.images = get_path_names(root_dir)
        self.classes = get_classes(root_dir)  # Get the list of class labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path)
        img = self.transform(img)

        print(f"Image Path: {img_path}")
        print(f"Label: {label}")

        return img, encode_label(label, classes_list=self.classes)

dataset = myDataset(root_dir="Dataset JPG", transform=transforms.ToTensor())
len(dataset)



class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        # Simulasi transformasi Resize
        if isinstance(data, str):
            return f"Resized '{data}' to {self.size}"
        elif isinstance(data, torch.Tensor):
            return f"Resized tensor with shape {data.shape} to {self.size}"
        else:
            raise ValueError("Data harus berupa string atau tensor pytorch.")

class RandomCropTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        # Simulasi transformasi RandomCrop
        if isinstance(data, str):
            return f"Random crop '{data}' to {self.size}"
        elif isinstance(data, torch.Tensor):
            return f"Random crop tensor with shape {data.shape} to {self.size}"
        else:
            raise ValueError("Data harus berupa string atau tensor pytorch.")


class RandomHorizontalFlipTransform:
    def __call__(self, data):
        # Simulasi transformasi RandomHorizontalFlip
        return f'Random horizontal flip'

class RandomRotationTransform:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, data):
        # Simulasi transformasi RandomRotation
        return f'Random rotate by {self.degrees} degrees'

class ToTensorTransform:
    def __call__(self, data):
        # Simulasi transformasi ToTensor
        return torch.tensor(data)

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # Simulasi transformasi Normalize
        return f'Normalized with mean={self.mean}, std={self.std}'

transform = Compose([
    ResizeTransform(128),
    RandomCropTransform(128),
    RandomHorizontalFlipTransform(),
    RandomRotationTransform(2),
    ToTensorTransform(),
    NormalizeTransform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])