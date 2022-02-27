from PIL import Image
import torch
from torch.utils.data import Dataset , DataLoader
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

#-----HYPER_PARAMTERS--------
batch_size = 16
phases = ['train', 'val', 'test']
class_to_int = {'cat' : 0, 'dog':1}

#-----LOCAL PATHS----------
TRAIN_DIR = 'dogs-vs-cats/train'
TEST_DIR = 'dogs-vs-cats/test1'

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

#Dataset that transforms, and returns based on the mode
class CatDogDataset(Dataset):
    def __init__(self, imgs, class_to_int, mode = 'train', transforms = None) -> None:
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index):
        image_name = self.imgs[index]

        if self.mode == 'train' or self.mode == 'val':
            image = Image.open(TRAIN_DIR + '/' + image_name)
            image = image.resize((224, 224))

            label = self.class_to_int[image_name.split('.')[0]]
            label = torch.tensor(label, dtype=torch.float32)

            image = self.transforms(image)

            return image, label

        elif self.mode == 'test':            
            image = Image.open(TEST_DIR + '/' + image_name)
            image = image.resize((224, 224))
            image = self.transforms(image)
            return image
    
    def __len__(self):
        return len(self.imgs)

print('Dataset class created')

#------ SPLITTING INTO THE TRAINING AND VALIDATION SETS-------------
images = os.listdir(TRAIN_DIR)
test_images = os.listdir(TEST_DIR)

train_images, val_images = train_test_split(images, test_size = 0.25)

print('Train-val split done')

#------ DATASET PREPARATION PRE-TRAINING ------------

image_datasets = {
    'train' : CatDogDataset(train_images, class_to_int, mode = 'train', transforms=data_transforms['train']),
    'val' : CatDogDataset(val_images, class_to_int, mode = 'val', transforms=data_transforms['val']),
    'test' : CatDogDataset(test_images, class_to_int, mode = 'test', transforms=data_transforms['val'])
}
dataloaders = {
        x : DataLoader(dataset=image_datasets[x], batch_size=8,shuffle=True)
        for x in phases
}
dataset_sizes = {x:len(image_datasets[x])
        for x in phases}
print('Dataloaders created!')

#-------VISUALIZATION---------

def visualize(image):
    image = image.numpy().transpose((1, 2, 0))
    image = image * std + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.show()

dataiter = iter(dataloaders['train'])
input, classes = dataiter.next()
out = torchvision.utils.make_grid(input)
#visualize(out)