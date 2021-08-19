"""
Author: Heidi Dye
Date: 6/29/2021
Version: 1
Purpose: Utilyize PyTorch to classify written digits 0-9 from the MNIST Dataset 
"""

#Import Libraries Here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose




#--------------------------------------#
#           CREATE THE MODEL           #
#--------------------------------------#

#Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
           nn.Linear(28*28, 512),
           nn.ReLU(),
           nn.Linear(512, 512),
           nn.ReLU(),
           nn.Linear(512, 10),
           nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    



#---------------------------------#
#          TRAIN AND TEST         #
#---------------------------------#

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        #Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
            
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


#-----------------------------#
#       GET THE DATASET       #
#-----------------------------#

#Download the training data from the open MNIST Dataset
training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    )

#Download the test data from the open MNIST Dataset
test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    )


#training_data, validation = random_split(training_data, [50000, 10000])
batch_size = 16
#Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
#valid_dataloader = DataLoader(validation, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork().to(device)
#print(model)

#loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = .001
weight_decay = [.0001]
prev_loss = 100
curr_loss = 100
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay[-1])
epochs = 0
while(prev_loss >= curr_loss):
    print(f"Epoch {epochs+1}\n-----------------------------------")
    prev_loss = curr_loss
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay[-1])
    weight_decay.append(weight_decay[-1]*10)
    train(train_dataloader, model, loss_fn, optimizer, device)
    curr_loss = test(test_dataloader, model, loss_fn, device)
    
    print(f"Weight Decay Adjusting: {weight_decay[-3]:>5f}")
    print(f"previous loss: {prev_loss:>8f}")
    print(f"current loss: {curr_loss:>8f}\n")
    
    epochs += 1

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay[-3])
prev_loss = curr_loss
        
while(prev_loss >= curr_loss):
    print(f"Epoch {epochs+1}\n-----------------------------------")
    prev_loss = curr_loss
    train(train_dataloader, model, loss_fn, optimizer, device)
    curr_loss = test(test_dataloader, model, loss_fn, device)
    print(f"Epoch Adjusting: {epochs+1}")
    print(f"previous loss: {prev_loss:>8f}")
    print(f"current loss: {curr_loss:>8f}\n ")
    epochs += 1

print("Done!")


'''
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
    print("Done!")
'''