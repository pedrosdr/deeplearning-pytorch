import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import KFold, cross_val_score

base = pd.read_csv('../data/iris.csv')

x = torch.tensor(base.iloc[:,:4].to_numpy(), dtype=torch.float32)
y = torch.tensor(pd.get_dummies(base.iloc[:,4:5], dtype='float32').to_numpy(), dtype=torch.float32)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

train_dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.dense1 = nn.Linear(4, 16)
        self.dense2 = nn.Linear(16, 16)
        self.dense3 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = self.dense1(x)
        x = f.relu(x)
        x = f.dropout(x, 0.2)
        
        x = self.dense2(x)
        x = f.relu(x)
        x = f.dropout(x, 0.2)
        
        x = self.dense3(x)
        x = f.softmax(x, dim=1)
        return x
        
net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    losses = []
    for (inputs, targets) in train_dataloader:
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    print(f'epoch {epoch}: loss={np.array(losses).mean()}')
    
ypred = net(xtest).detach()
ypred = np.array([_.argmax() for _ in ypred])
ytrue = np.array([_.argmax() for _ in ytest])
accuracy_score(ytrue, ypred)
