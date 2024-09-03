import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(123)
torch.manual_seed(123)

x = pd.read_csv('../data/entradas_breast.csv')
y = pd.read_csv('../data/saidas_breast.csv')

xtrain, ytrain, xtest, ytest = train_test_split(x,y, test_size=0.25)
