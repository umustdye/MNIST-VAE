"""
Author: Heidi Dye
Date: 
Version: 1.0
Purpose: Convolutional Neural Network with the MNIST Dataset
"""


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as functional

import matplotlib.pyplot as plt
import numpy as np




#--------------------------------------#
#           CREATE THE MODEL           #
#--------------------------------------#

#Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Using {} device".format(device))

#define the model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 12, 5)
        self.layer1 = nn.Linear(192, 250)
        self.layer2 = nn.Linear(250, 100)
        self.layer3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(functional.relu(self.conv2(x)))
        #print(x.shape)
        #flatten all dimensions except the batch
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = functional.relu(self.layer1(x))
        #print(x.shape)
        x = functional.relu(self.layer2(x))
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        return x
    


#-----------------------------#
#       GET THE DATASET       #
#-----------------------------#

#transform to tensors from range [0, 1] to a normalized range of [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

#Download the training data from the open MNIST Dataset
training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform
    )

#Download the test data from the open MNIST Dataset
test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform
    )

batch_size = 4

#Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

#tuple for the possible classifcations for image output
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


#------------------------------------#
#           SHOW THE IMAGES          #
#------------------------------------#

def showImage(img):
    #unnormalize
    img = img/2 + 0.5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

#show images
showImage(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


#---------------------------------#
#          TRAIN AND TEST         #
#---------------------------------#

def train(dataloader, model, loss_fn, optimizer, device):
    running_loss = 0.0
    for batch, (inputs, labels) in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch % 2000 == 1999:    # print every 2000 mini-batches
            print('loss: %.3f' %(running_loss / 2000))
            running_loss = 0.0
            

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

model = CNN().to(device)
#loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = .001
momentum = .9
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)


epochs = 2

for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
    print("Done!")
