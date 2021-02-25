# Author: Andrew Bertino
# Credit to Python Engineer on YouTube for the wonderful PyTorch walkthrough this is based off of.
# Video: https://www.youtube.com/watch?v=oPhxf2fXHkQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=13

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device config, always cpu for me because i have AMD card
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#HYPER PARAMETERS
#images from MNIST are 28X28, or 784 after flattening
input_size = 784 
#number of hidden layer nodes
hidden_size = 600
#digits from 0-9
num_classes = 10
#number of training loops
num_epochs = 5
#how many samples per batch to load
batch_size = 100
#how drastically to react when learning
learning_rate = 0.003

#MNIST data set
dataset = torchvision.datasets.MNIST(root='./data',train=True, transform = transforms.ToTensor(),download=True)

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55000,5000])
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #first linear layer
        self.l1 = nn.Linear(input_size, hidden_size)
        #activation function ReLU
        self.relu = nn.ReLU()
        #second linear layer
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#loss and optimizer(applies "softmax" to squish output between 0 and 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### TRAINING LOOP ###
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    n_correct = 0
    n_samples = 0
    for i, (images,labels) in enumerate(train_loader):
        #100,1,28,28 => 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        #add +1 for each correct prediction
        n_correct += (predictions == labels).sum().item()
    
        #backward
        optimizer.zero_grad()
        loss.backward()

        #algorithm to update parameters
        optimizer.step()

        #print information every x steps
        if (i+1) % 55 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
   
    acc = 100*n_correct/n_samples
    print(f'Training accuracy = {acc}')       
   
 
### VALIDATION LOOP ###
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in validation_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        #value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        #add +1 for each correct prediction
        n_correct += (predictions == labels).sum().item()
    
    acc = 100*n_correct/n_samples
    print(f'Validation accuracy = {acc}')

input("Change hyperparameters, or continue with current settings")


### TESTING LOOP ###
#dont want to compute gradients when testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        #value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        #add +1 for each correct prediction
        n_correct += (predictions == labels).sum().item()
    
    acc = 100*n_correct/n_samples
    print(f'Testing accuracy = {acc}')


#validation loop?
