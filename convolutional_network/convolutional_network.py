import torch
import torch.nn as nn
import torch.nn.functional as f 
import numpy as np
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0')

transform = transforms.ToTensor()
train = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test = datasets.MNIST(root='.', train=False, download=True, transform=transform)

dataloader_train = torch.utils.data.DataLoader(train, batch_size=1000)
dataloader_test = torch.utils.data.DataLoader(test, batch_size=1000)

image = next(iter(dataloader_train))[0][9].view(28, 28, 1)
print(image)
plt.imshow(image, cmap='gray')

# Building the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, [3,3], padding='same')
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, [3,3], padding='same')
        self.bnorm2 = nn.BatchNorm2d(32)
        self.dense1 = nn.Linear(1568, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.bnorm1(x)
        
        x = f.max_pool2d(x, [2,2])
        
        x = self.conv2(x)
        x = f.relu(x)
        x = self.bnorm2(x)
        
        x = f.max_pool2d(x, [2,2])
        
        x = x.view(x.shape[0], 32 * 7 * 7)
        x = self.dense1(x)
        x = f.softmax(x, dim=1)
        return x  
    
net = Net()
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    losses = []
    for inputs, targets in dataloader_train:
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    losses = np.array(losses)
    print(f'{losses.mean():.4f}')

y_true = [_[1] for _ in test]
xtest = torch.tensor(np.array([_[0] for _ in test]), dtype=torch.float32)
xtest = xtest.to(device)
y_pred = net(xtest)
y_pred = y_pred.to(torch.device('cpu')).detach().numpy().argmax(axis=1)

accuracy_score(y_true, y_pred)
