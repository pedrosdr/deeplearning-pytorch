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
from torchvision import datasets, models
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

# Loading the model VGG 16
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
print(model)

# Replacing the output layer
n_inputs = model.classifier[6].in_features
classification_layer = nn.Linear(n_inputs, len(train_dataset.classes))
model.classifier[6] = classification_layer

# freezing the weights of the convolutional layers 
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())

model = model.to(device)
running_loss = 0
running_accuracy = 0
for epoch in range(10):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        y_pred = torch.argmax(f.softmax(outputs, dim=1), dim=1)
        equals = (y_pred == targets).float()
        accuracy = equals.mean()
        
        running_loss += loss.item()
        running_accuracy += accuracy
        
        print(f'epoch: {epoch+1}, step {i}/{len(train_loader)}, accuracy: {accuracy}, loss: {loss.item()}')
        
# Testing
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

outputs = model(images)
y_pred = torch.argmax(f.softmax(outputs, dim=1), dim=1)
equals = (y_pred == labels).float()
accuracy = equals.mean()
print(accuracy)

idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
labels, y_pred = labels.to(cpu).detach().numpy(), y_pred.to(cpu).detach().numpy()
labels = [idx_to_class[label] for label in labels]
y_pred = [idx_to_class[prediction] for prediction in y_pred]
