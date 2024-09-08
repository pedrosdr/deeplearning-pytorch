import torch
import torch.nn as nn
import torch.nn.functional as f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda')
cpu = torch.device('cpu')

base1 = pd.read_csv('../data/petr4_treinamento.csv')
base2 = pd.read_csv('../data/petr4_teste.csv')
base = pd.concat([base1, base2], axis=0)
base = base.dropna()
base_np = base.to_numpy()
base_np = base_np[:,1:]

basetrain = base_np[:1017,:]
basetest = base_np[1017:,:]

scaler = MinMaxScaler()
basetrain = scaler.fit_transform(basetrain)
basetest = scaler.transform(basetest)

base_np = np.concatenate([basetrain, basetest], axis=0)

# creating the TimeSeriesTransformer
class TimeSeriesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_periods):
        self.n_periods = n_periods

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        x_arr = []
        y_arr = []

        for i in range(self.n_periods, X.shape[0]):
            x_arr.append(X[i-self.n_periods:i,:])
            if y is not None:
                y_arr.append(y[i,:])

        if y is not None:
            return(np.array(x_arr), np.array(y_arr))
        else:
            return np.array(x_arr)

x, y = TimeSeriesTransformer(10).transform(base_np, base_np[:,3:4])

xtrain = x[:1007,:]
ytrain = y[:1007,:]
xtest = x[1007:,:]
ytest = y[1007:,:]

xtrain = torch.tensor(xtrain, dtype=torch.float32)
xtest = torch.tensor(xtest, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

next(iter(dataloader))

# Building the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm1 = nn.LSTM(6, 100)
        self.lstm2 = nn.LSTM(100, 50)
        self.lstm3 = nn.LSTM(50, 50, 2)
        self.dense1 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = x.permute(1,0,2)
        x, (hn, cn) = self.lstm1(x)
        x = f.dropout(x)
        
        x, (hn, cn) = self.lstm2(x)
        x = f.dropout(x)
        
        x, (hn, cn) = self.lstm3(x)
        x = f.dropout(x)
        
        x = x[-1]
        x = self.dense1(x)
        return x

net = Net().to(device)

optimizer = torch.optim.Adam(net.parameters(), 0.001, weight_decay=0.0001)
criterion = nn.MSELoss()

for i in range(1000):
    losses = []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    losses = np.array(losses)
    print(f'epoch {i}, loss: {losses.mean():.3f}')

# Testing the model
ypred = net(xtest.to(device)).to(cpu).detach().numpy()
plt.plot(list(range(len(ypred))), ypred)
plt.plot(list(range(len(ypred))), ytest)
