import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models

import numpy as np
import matplotlib.pyplot as plt

from dataHandling import dataloaders, dataset_sizes
from trainer import trainModel2
from modelTester import model_accuracy

print(dataset_sizes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----HYPER_PARAMTERS--------
num_epochs = 15
learning_rate = 0.001

def accuracy(preds, truths):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == truths[i] else 0 for i in range(len(preds))]

    acc = np.sum(acc) / len(preds)

    return (acc * 100.0)

#------MODEL-------

model_conv = models.alexnet(pretrained=True)

#print(model_conv)

model_conv.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=1, bias=True),
    nn.Sigmoid()
)

model_conv = model_conv.to(device)

# Modifying Head - classifier

criterion = nn.BCELoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=learning_rate, momentum=0.9)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_conv, step_size=7, gamma=0.1)

model_conv = trainModel2(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs, dataloaders=dataloaders, file_used='alexnet_best.pth')

#acc = model_accuracy(model_conv)
#print(f'Test set accuracy: {acc}')