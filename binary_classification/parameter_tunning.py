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
from sklearn.model_selection import GridSearchCV

np.random.seed(123)
torch.manual_seed(123)

x = torch.tensor(pd.read_csv('../data/entradas_breast.csv').to_numpy(), dtype=torch.float32)
y = torch.tensor(pd.read_csv('../data/saidas_breast.csv').to_numpy(), dtype=torch.float32)

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25)

class Net(nn.Module):
    def __init__(self, activation, neurons):
        super(Net, self).__init__()
        self.activation = activation
        
        self.fc1 = nn.Linear(30, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = f.dropout(x, 0.2)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = f.dropout(x, 0.2)
        
        x = self.fc3(x)
        x = f.sigmoid(x)
        
        return x

net = Net(f.relu, 16)
sk_net = NeuralNetBinaryClassifier(
    module=net,
    lr=0.001,
    max_epochs=100,
    batch_size=100,
    train_split=False,
    criterion=nn.BCELoss
)

params = {
    'optimizer': [torch.optim.Adam, torch.optim.RMSprop],
    'module__activation': [f.relu, f.sigmoid],
    'module__neurons': [16, 50, 100]
}

gs = GridSearchCV(sk_net, params, scoring='accuracy', cv=5)
gs.fit(x, y.squeeze())
gs.best_params_
gs.best_score_

sk_net.fit(xtrain, ytrain.squeeze())

# Training the best model
dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

model = Net(f.sigmoid, 100)
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(1000):
    losses = []
    for (inputs, targets) in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    losses = np.array(losses).mean()
    print(f'epoch {epoch}: bce={losses}')
    
ypred = model(x).detach()
ypred = [1 if _ > 0.5 else 0 for _ in ypred.flatten()]
ytrue = y.detach()
accuracy_score(ytrue, ypred)
        
torch.save(model.state_dict(), 'binary_classifier.pth')

# Loading the classifier
x = torch.tensor(pd.read_csv('../data/entradas_breast.csv').to_numpy(), dtype=torch.float32)
y = torch.tensor(pd.read_csv('../data/saidas_breast.csv').to_numpy(), dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, activation, neurons):
        super(Net, self).__init__()
        self.activation = activation
        
        self.fc1 = nn.Linear(30, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = f.dropout(x, 0.2)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = f.dropout(x, 0.2)
        
        x = self.fc3(x)
        x = f.sigmoid(x)
        
        return x
    
model = Net(f.sigmoid, 100)
model.load_state_dict(torch.load('binary_classifier.pth'))

ypred = model(x).detach()
ypred = [1 if _ > 0.5 else 0 for _ in ypred.flatten()]
ytrue = y.detach()
accuracy_score(ytrue, ypred)
