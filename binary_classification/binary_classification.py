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


for epoch in range(100):
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


# Viewing the weights
params = list(net.parameters())
print(net.fc1.weight is params[0])
print(net.fc1.bias is params[1])

for param in params:
    print(param.shape)
    sns.histplot(param.detach().numpy().flatten())
    plt.show()
    plt.close()
    

# Avaliação do modelo
ypred = torch.tensor([1 if x > 0.5 else 0 for x in net(xtest)], dtype=torch.int32)
ypred = ypred.unsqueeze(-1)

print(accuracy_score(ytest, ypred))

# Sktorch
sk_net = NeuralNetBinaryClassifier(
    module=net,
    optimizer=torch.optim.Adam,
    lr=0.001,
    optimizer__weight_decay=0.0001,
    max_epochs=500,
    batch_size=100,
    train_split=False
)

# Cross Validation
kf = KFold(n_splits=10, shuffle=True)
x = torch.concat([xtrain, xtest], dim=0)
y = torch.concat([ytrain, ytest], dim=0).squeeze()
cv_results = cross_val_score(sk_net, x, y, cv = kf, scoring='accuracy')
sns.histplot(cv_results, bins=7)
