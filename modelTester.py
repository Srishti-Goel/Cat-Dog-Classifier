import enum
import numpy as np
import torch
from dataHandling import dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(preds, truths):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == truths[i] else 0 for i in range(len(preds))]

    acc = np.sum(acc) / len(preds)

    return (acc * 100.0)

def model_accuracy(model):
    accs = []
    for i, (images, labels) in enumerate(dataloaders['train']):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        acc = accuracy(outputs, labels)

        accs.append(acc)

    for i, (images, labels) in enumerate(dataloaders['train']):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        acc = accuracy(outputs, labels)

        accs.append(acc)
    
    acc = np.mean(accs)
    return acc
