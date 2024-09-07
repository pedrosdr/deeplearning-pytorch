import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets
import matplotlib.pyplot as plt

device = torch.device('cuda')

base_train = datasets.CIFAR10(root='.', train=True, download=True)
base_test = datasets.CIFAR10(root='.', train=False)

xtrain = np.array([np.array(image) for image, _ in base_train])
ytrain = np.array([np.array(target) for _, target in base_train])
xtrain = torch.tensor(xtrain, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.float32)
xtrain = xtrain.view(xtrain.shape[0], 3, 32, 32)
ytrain = ytrain.view(-1, 1)

xtest = np.array([np.array(image) for image, _ in base_test])
ytest = np.array([np.array(target) for _, target in base_test])
xtest = torch.tensor(xtest, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.float32)
xtest = xtest.view(xtest.shape[0], 3, 32, 32)
ytest = ytest.view(-1, 1)

xtrain = xtrain/255
xtest = xtest/255

dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)


# Creating model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, [3,3], padding='same')
        self.bnorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 6, [3,3], padding='same')
        self.bnorm2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 12, [3,3], padding='same')
        self.bnorm3 = nn.BatchNorm2d(12)
        
        self.dense1 = nn.Linear(192, 96)
        self.dense2 = nn.Linear(96, 96)
        
        self.dense3 = nn.Linear(96, 96)
        self.dense4 = nn.Linear(96, 192)
        
        self.conv4 = nn.ConvTranspose2d(12, 6, [2,2], stride=[2,2])
        self.bnorm4 = nn.BatchNorm2d(6)
        self.conv5 = nn.ConvTranspose2d(6, 6, [2,2], stride=[2,2])
        self.bnorm5 = nn.BatchNorm2d(6)
        self.conv6 = nn.ConvTranspose2d(6, 3, [2,2], stride=[2,2])

        
    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.bnorm1(x)
        
        x = f.max_pool2d(x, [2,2])
        
        x = self.conv2(x)
        x = f.relu(x)
        x = self.bnorm2(x)
        
        x = f.max_pool2d(x, [2,2])
        
        x = self.conv3(x)
        x = f.relu(x)
        x = self.bnorm3(x)
        
        x = f.max_pool2d(x, [2,2])
        
        x = x.view(x.shape[0], 192)
        
        x = self.dense1(x)
        x = f.relu(x)
        
        x = self.dense2(x)
        x = f.relu(x)
        
        x = self.dense3(x)
        x = f.relu(x)
        
        x = self.dense4(x)
        x = f.relu(x)
        
        x = x.view(x.shape[0], 12, 4, 4)
        
        x = self.conv4(x)
        x = f.relu(x)
        x = self.bnorm4(x)
        
        x = self.conv5(x)
        x = f.relu(x)
        x = self.bnorm5(x)
        
        x = self.conv6(x)
        x = f.sigmoid(x)
        return x
        
net = Net().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50000):
    inputs = image = xtrain[1:2,:]
    inputs = inputs.to(device)
    
    optimizer.zero_grad()
    outputs = net(inputs)
    outputs = outputs.to(device)
    
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()
        
    print(f'epoch: {epoch}, loss: {loss.item()}')

image = xtrain[1:2,:]
image = image.to(device)
output = net(image)
output = output.to(torch.device('cpu')).detach().numpy().reshape(32,32,3)
image = image.to(torch.device('cpu')).detach().numpy().reshape(32,32,3)
plt.imshow(output)
plt.imshow(image)
