import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

base = pd.read_csv('../data/games.csv')

base.Platform = np.where(
    base.Platform.isin(['GG','PCFX','TG16','3DO','WS','SCD','NG','GEN','DC',
     'GB','NES','2600','WiiU','SAT','SNES','XOne','N64','PSV','3DS','GC', 'GBA','XB']),
                        'Other', base.Platform
)
base.Rating = base.Rating.map({
    np.nan: np.nan,
    'RP': np.nan,
    'EC': 0,
    'E': 0,
    'K-A': 0,
    'E10+': 1,
    'T': 2,
    'M': 3,
    'AO': 3    
})

cate_cols = ['Platform', 'Genre']

base.User_Score = pd.to_numeric(base.User_Score, errors='coerce')
num_cols = ['Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating']

# Dividing the base
base_train, base_test = train_test_split(base, test_size=0.25)

# Pipelines
pipe_cate = Pipeline(steps=[
   ('imputer', SimpleImputer(strategy='most_frequent')),
   ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

pipe_numeric = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

transformer = ColumnTransformer(transformers=[
    ('categorical', pipe_cate, cate_cols),
    ('numeric', pipe_numeric, num_cols)
])

xtrain = transformer.fit_transform(base_train)
xtrain = xtrain.toarray()

xtest = transformer.transform(base_test)
xtest = xtest.toarray()

# Base test
ytrain = base_train.loc[:,['NA_Sales', 'EU_Sales', 'JP_Sales']]
ytest = base_test.loc[:,['NA_Sales', 'EU_Sales', 'JP_Sales']]

pipe_y = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())    
])

ytrain = pipe_y.fit_transform(ytrain)
ytest = pipe_y.transform(ytest)

pipe_y[1].inverse_transform(ytest)

# Converting to tensor
xtrain = torch.tensor(xtrain, dtype=torch.float32)
xtest = torch.tensor(xtest, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.float32)

# Building the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.dense1 = nn.Linear(28, 14)
        self.dense2 = nn.Linear(14, 14)
        self.dense3 = nn.Linear(14, 3)
        
        self.compiled = False
        self.optimizer = None
        self.criterion = None
        
    def compile(
            self, 
            optimizer=None, 
            criterion=nn.MSELoss()):
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        else:    
            self.optimizer = optimizer
        self.criterion = criterion
        self.compiled = True
        
    def forward(self, x):
        x = self.dense1(x)
        x = f.relu(x)
        x = f.dropout(x, 0.2)
        
        x = self.dense2(x)
        x = f.relu(x)
        x = f.dropout(x, 0.2)
        
        x = self.dense3(x)
        return x

    def fit(self, x, y, epochs=1, batch_size=100):
        if not self.compiled:
            raise ValueError('Model not compiled.')
            
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        for epoch in range(epochs):
            losses = []
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            
            losses = np.array(losses)
            print(f'epoch {epoch}, loss: {losses.mean():.2f}')

net = Net()
net.compile()
net.fit(xtrain, ytrain, 1000, 1000)

y_pred = net(xtest)
y_pred = y_pred.detach().numpy()
y_pred = pipe_y[1].inverse_transform(y_pred)

y_true = ytest.detach().numpy()
y_true = pipe_y[1].inverse_transform(y_true)
