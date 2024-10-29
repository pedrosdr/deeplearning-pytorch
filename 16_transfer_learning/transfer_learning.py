import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# directories_info = pd.DataFrame({
#     'directory': [dirs for root, dirs, files in os.walk('Images')][0],
#     'number_of_files': [len(files) for root, dirs, files in os.walk('Images')][1:]  
# })

# directories_info = directories_info.sort_values('number_of_files', axis=0)

# sns.lineplot(
#     data=directories_info, 
#     x=range(len(directories_info)), 
#     y='number_of_files'
# )

# directories_filtered = directories_info[
#     directories_info['number_of_files'] >= 600    
# ]

# directories_to_remove = [x for x in directories_info[
#     ~directories_info['directory'].isin(directories_filtered['directory'])
#    ]['directory']
# ]

# [shutil.rmtree('Images/' + directory) for directory in directories_to_remove]

# df = pd.DataFrame()

# dirs = [root for root, dirs, files in os.walk('Images') if len(files) != 0]


# for d in dirs:
#     files = os.listdir(d)
#     ftrain, ftest = train_test_split(files, test_size=0.2, random_state=23)
    
#     folder_name = d.split('\\')[1]
#     from_dir = d
#     to_dir_train = os.path.join('train', folder_name)
#     to_dir_test = os.path.join('test', folder_name)
    
#     os.makedirs(to_dir_train, exist_ok=True)
#     os.makedirs(to_dir_test, exist_ok=True)
    
#     for f in ftrain:
#         shutil.copy(
#             os.path.join(from_dir, f),
#             os.path.join(to_dir_train, f)
#         )
        
#     for f in ftest:
#         shutil.copy(
#             os.path.join(from_dir, f),
#             os.path.join(to_dir_test, f)
#         )


import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda')
cpu = torch.device('cpu')

torch.manual_seed(7)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()    
])

train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
plt.imshow(train_dataset[30][0].permute(1,2,0))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataset = datasets.ImageFolder('dataset/test', transform=transform)
plt.imshow(test_dataset[30][0].permute(1,2,0))

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)












