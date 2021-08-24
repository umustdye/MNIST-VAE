"""
Author: Heidi Dye
Date: 7/20/2021
Version: 2.0
Purpose: Variational Autoencoder(VAE) with the MINST Dataset using PyTorch
"""

#Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np


#ignore warnings, i.e those annoying ones that appear for every run
import warnings
warnings.filterwarnings("ignore")


#-------------------------------------#
#           CREATE THE VAE            #
#-------------------------------------#
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), #16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), #64, 1, 1
            nn.ReLU(),
            Compress()
            )
        
        
        self.fc_mu = nn.Linear(64, 64)
        self.fc_logvar = nn.Linear(64, 64)
        
        
        
        #64, 1, 1
        self.decoder = nn.Sequential(
            Decompress(),
            nn.ConvTranspose2d(64, 32, 7), #32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), #1, 28, 28
            nn.Sigmoid()
            )
        
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        z = mu + (std * e)
        return z
    
    def generateExample(self, num_examples):
        #x must be 64, 1, 1 tensor
        mu = torch.zeros(num_examples, 64).to("cuda")
        logvar = torch.ones(num_examples, 64).to("cuda")
        z = self.sample(mu, logvar)
        
        return self.decoder(z)
    
    def forward(self, x):
        #print(f"Before encoding: {x.shape}")
        x = self.encoder(x)
        #print(f"After encoding: {x.shape}")
        mu = self.fc_mu(x)
        #mu = torch.zeros(10, 64).to("cuda")
        #print(f"Mean (mu) Shape: {mu.shape}")
        logvar = self.fc_logvar(x)
        #logvar = torch.ones(10, 64).to("cuda")
        #print(f"Log Variance Shape: {logvar.shape}")
        z = self.sample(mu, logvar)
        #print(f"Z Shape: {z.shape}")
        recon_x = self.decoder(z)
        #mu 0 logvar 1 for recon
        #print(f"Reconstructed x After Decoding: {recon_x.shape}")
        return recon_x, mu, logvar

class Compress(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Decompress(nn.Module):
    def forward(self, input, size=28*28):
        #print(input.view(input.size(0), input.size(1), 1, 1).shape)
        return input.view(input.size(0), input.size(1), 1, 1)

def loss_fn(recon_x, x, mu, logvar):
    #use squared error MSELoss
    loss = functional.mse_loss(recon_x, x, reduction="sum")
    #BCELoss = functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KL_Div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss + KL_Div
    #return BCELoss + KL_Div



    
#-----------------------------#
#       GET THE DATASET       #
#-----------------------------#

#transform to tensors from range [0, 1] to a normalized range of [-1, 1], NVM
transform = transforms.ToTensor()

#Download the dataset from the open MNIST Dataset
dataset = datasets.MNIST(
    root = "data",
    download = True,
    transform = transform
    )

batch_size = 10

#Create data loaders
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#------------------------------------#
#           SHOW THE IMAGES          #
#------------------------------------#

def showImage(img, img_recon, epoch):
    #unnormalize
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    #img = img/2 + 0.5
    img = img.numpy()
    plt.title(label="Original Epoch: #"+str(epoch))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    fig.add_subplot(1, 2, 2)
    #img_recon = img_recon/2 + 0.5
    img_recon = img_recon.numpy()
    plt.title(label="Reconstruction Epoch: #"+str(epoch))
    plt.imshow(np.transpose(img_recon, (1, 2, 0)))
    plt.show(block=True)
    
def showExample(img):
    #unnormalize
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    #img = img/2 + 0.5
    img = img.numpy()
    plt.title(label="Generated Example")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show(block=True)
    
#dataiter = iter(train_dataloader)
#images, labels = dataiter.next()

#show images
#showImage(torchvision.utils.make_grid(images))


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}\n")



#training the VAE initially

#one image channel because black and white
model = VAE().to(device)

learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 500
running_loss = 0.0


for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    for idx, (images, _) in enumerate(train_dataloader):
        #running_loss = 0.0
        images = images.to(device)
        recon_x, mu, logvar = model(images)
        loss = loss_fn(recon_x, images, mu, logvar)
         # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        
        # print statistics
        running_loss += loss.item()
        if idx % 2000 == 1999:    # print every 2000 mini-batches
            #output the original image and the reconstructed image
            showImage(torchvision.utils.make_grid(images.to("cpu")), torchvision.utils.make_grid(recon_x.to("cpu")), t+1)
            #showImage(torchvision.utils.make_grid(recon_x.to("cpu")))
            print('loss: %.3f' %(running_loss / 2000))
            running_loss = 0.0
    #print(f"Done with epoch #{t+1}\n")
    
  
    


'''
#save the trained model
print("Saving model...")
torch.save(model.state_dict(), "MNIST_VAE_MODEL_2.pt")
print("Model saved.")
'''

'''
#For reference---------
Save:
torch.save(model.state_dict(), PATH)

Load:
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
-----------------------
'''
'''
#load the saved model (500 epoches)
print("Loading previously saved model..")
model = VAE().to(device)

model.load_state_dict(torch.load("MNIST_VAE_MODEL.pt"))
#print(model)
model.eval()
print("Model has been loaded.")
'''

'''
print("Loaded model's weight")
for param in model.parameters():
  print(param.data)
'''
  
#generate a random example
print("Generating a random example...")
num_examples = 10
#run the example through the decoder
example = model.generateExample(num_examples)
showExample(torchvision.utils.make_grid(example.to("cpu")))