import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(123)
torch.manual_seed(123)

x = torch.tensor(pd.read_csv('../data/entradas_breast.csv').to_numpy(), dtype=torch.float32)
y = torch.tensor(pd.read_csv('../data/saidas_breast.csv').to_numpy(), dtype=torch.float32)

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25)

dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        
        x = self.fc2(x)
        x = f.relu(x)
        
        x = self.fc3(x)
        x = f.sigmoid(x)
        
        return x

net = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), 0.001, weight_decay=0.0001)


for epoch in range(1000):
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        # loss = ((outputs - labels) ** 2.0).sum()
        loss.backward()
        optimizer.step()
        
        running_loss += loss
        
    running_loss /= len(train_loader)
    print(running_loss)

ypred = torch.tensor([1 if x > 0.5 else 0 for x in net(xtest)], dtype=torch.int32)
ypred = ypred.unsqueeze(-1)

print(accuracy_score(ytest, ypred))
