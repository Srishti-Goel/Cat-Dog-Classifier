import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(preds, truths):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == truths[i] else 0 for i in range(len(preds))]

    acc = np.sum(acc) / len(preds)

    return (acc * 100.0)

def trainModel2(model, criterion, optimizer, scheduler, num_epochs, dataloaders, file_used = 'best_cat_dog.pth'):
    training_start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses_over_time = []
    accs_over_time = []

    for epoch in range(num_epochs):
        print(f'\n Epoch: {epoch + 1}/{num_epochs} --------')
        epoch_start = time.time()


        for phase in ['train', 'val']:
            
            epoch_loss = []
            epoch_acc = []

            if(phase == 'train'):
                model.train()
            else:
                model.eval()

            n_phases = len(dataloaders[phase])
            print(f"No. of phases: {n_phases}")
            start_phase = time.time()

            for i, (images, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                labels = labels.reshape((labels.shape[0], 1))
                preds = model(images)
                loss = criterion(preds, labels)

                epoch_loss.append(loss.item())
                epoch_acc.append(accuracy(preds, labels))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if i % 300 == 0:
                    print(f'{phase} {i+1}/{n_phases} | Loss: {loss :.4f} | Accuracy: {accuracy(preds, labels)}')
            
            end_phase = time.time()
            total_time = end_phase - start_phase

            epoch_loss = np.mean(epoch_loss)
            epoch_acc = np.mean(epoch_acc)

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), file_used)
                    best_model_weights = copy.deepcopy(model.state_dict())

                print("\nValidating over")
            
            else:
                print('\nTraining over')
            
        print(f'Epoch: {epoch + 1} | Loss : {epoch_loss:.4f} | Acc: {epoch_acc :.4f} | Time : {(time.time() - epoch_start) :.4f}')
        losses_over_time.append(epoch_loss)
        accs_over_time.append(epoch_acc)
        

    time_of_training = time.time() - training_start
    print(f'Training completed in: {time_of_training // 60:.0f}m {time_of_training % 60 :.2f}s')
    print(f'Best accuracy: {best_acc:.4f}')

    print(len(losses_over_time))

    plt.plot(range(len(losses_over_time)), losses_over_time)
    plt.show()

    model.load_state_dict(best_model_weights)
    return model