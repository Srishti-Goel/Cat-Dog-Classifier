import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models

import numpy as np
import matplotlib.pyplot as plt
import time
import copy

from modelTester import model_accuracy

from dataHandling import dataloaders, dataset_sizes
from trainer import trainModel2

print(dataset_sizes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----HYPER_PARAMTERS--------
num_epochs = 2
learning_rate = 0.001

def accuracy(preds, truths):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == truths[i] else 0 for i in range(len(preds))]

    acc = np.sum(acc) / len(preds)

    return (acc * 100.0)

def train_one_epoch(train_data_loader, optimizer, criterion, model):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for i, (images, labels) in enumerate(train_data_loader):
        if (i) % 100 == 0:
            print("I'm working")
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        _loss.backward()
        optimizer.step()
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
        
    return epoch_loss, epoch_acc, total_time

def val_one_epoch(val_data_loader, best_val_acc, model):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for i, (images, labels) in enumerate(val_data_loader):
        
        if i%100 == 0:
            print("I'm evaluating!")

        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),"resnet50_best.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc

model = models.resnet50(pretrained = True)

# Modifying Head - classifier

model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

#Loss Function
criterion = nn.BCELoss()

# Logs - Helpful for plotting after training finishes
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

# Loading model to device
model.to(device)

# No of epochs 
epochs = 5

best_val_acc = 0
for epoch in range(epochs):
 
    #Training
    loss, acc, _time = train_one_epoch(dataloaders['train'], optimizer, criterion, model)
    
    #Print Epoch Details
    print("\nTraining")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))
    
    ###Validation
    loss, acc, _time, best_val_acc = val_one_epoch(dataloaders['val'], best_val_acc, model)
    
    #Print Epoch Details
    print("\nValidating")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))


def trainModel(model, criterion, optimizer, scheduler, num_epochs = num_epochs):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}-----------')

        for phase in ['train', 'val']:
            if(phase == 'train'):
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            n_phases = len(dataloaders[phase])
            print(n_phases)
            start_time = time.time()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.reshape((labels.shape[0], 1))

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(labels, outputs)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                if (i+1) % 100 == 0:
                    print(f'Phase: {phase} | i: {i+1}/{n_phases} | sizes: {labels.size()} | Time: {time.time() - start_time} | Loss: {loss.item()}')
                    start_time = time.time()
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = accuracy(outputs, labels)

            print(f'{phase} | Loss: {epoch_loss} | Accuracy: {epoch_acc}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print(f'Training completed in: {time_elapsed // 60:.0f}m {time_elapsed % 60}s')
    print(f'Best accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_weights)
    return model

model_conv = models.resnet50(pretrained=True)

model_conv.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)

model_conv = model_conv.to(device)

model = models.resnet50(pretrained = True)

# Modifying Head - classifier

model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=learning_rate, momentum=0.9)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_conv, step_size=7, gamma=0.1)

model_conv = trainModel(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs, dataloaders=dataloaders)

#acc = model_accuracy(model_conv)
#print(f'Test set accuracy: {acc}')