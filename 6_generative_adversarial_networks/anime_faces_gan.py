import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

files = np.array(['../data/anime_faces/'+_ for _ in os.listdir('../data/anime_faces')])
indexes = np.random.randint(0, len(files), 3000)

images = np.array([np.array(Image.open(_).resize((64,64))) for _ in files[indexes]])
del files

images = torch.tensor(images, dtype=torch.float32)
images = images.permute(0,3,1,2)
images = images / 255.0

dataset = torch.utils.data.TensorDataset(images)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense1 = nn.Linear(300, 256 * 8 * 8)
        
        self.convt1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bnorm1 = nn.BatchNorm2d(128)
        
        self.convt2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bnorm2 = nn.BatchNorm2d(64)
        
        self.convt3 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.bnorm3 = nn.BatchNorm2d(32)
        
        self.convt4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.bnorm4 = nn.BatchNorm2d(16)
        
        self.convt5 = nn.ConvTranspose2d(16, 3, 3, 1, 1)
        
    def forward(self, x):
        x = f.dropout(f.relu(self.dense1(x)), 0.2)
        x = x.view(-1, 256, 8, 8)
        x = f.dropout(self.bnorm1(f.relu(self.convt1(x))), 0.2)
        x = f.dropout(self.bnorm2(f.relu(self.convt2(x))), 0.2)
        x = f.dropout(self.bnorm3(f.relu(self.convt3(x))), 0.2)
        x = f.dropout(self.bnorm4(f.relu(self.convt4(x))), 0.2)
        x = f.sigmoid(self.convt5(x))
        return x
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, 2)
        self.bnorm1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.bnorm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 128, 3, 2)
        self.bnorm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 'same')
        self.bnorm4 = nn.BatchNorm2d(128)
        
        self.dense1 = nn.Linear(6272, 3136)
        self.dense2 = nn.Linear(3136, 1)
    
    def forward(self, x):
        x = f.dropout(self.bnorm1(f.leaky_relu(self.conv1(x), 0.2)), 0.2)
        x = f.dropout(self.bnorm2(f.leaky_relu(self.conv2(x), 0.2)), 0.2)
        x = f.dropout(self.bnorm3(f.leaky_relu(self.conv3(x), 0.2)), 0.2)
        x = f.dropout(self.bnorm4(f.leaky_relu(self.conv4(x), 0.2)), 0.2)

        x = x.view(-1, 6272)
        x = f.dropout(f.relu(self.dense1(x)), 0.2)
        x = f.sigmoid(self.dense2(x))
        return x

gen = Generator().to(device)
disc = Discriminator().to(device)

disc(torch.rand([1,3,64,64], dtype=torch.float32, device=device)).shape

optim1 = torch.optim.RMSprop(gen.parameters(), lr=0.001, weight_decay=0.0001)
optim2 = torch.optim.RMSprop(disc.parameters(), lr=0.001, weight_decay=0.0001)

criterion = nn.BCELoss()

for i in range(10):
    for [inputs] in dataloader:
        inputs = inputs.to(device)
        
        gen_outputs = gen(torch.randn([len(inputs), 300], dtype=torch.float32, device=device))
        
        target_true = torch.ones([len(inputs), 1], dtype=torch.float32, device=device)
        target_false = torch.zeros([len(inputs), 1], dtype=torch.float32, device=device)
        
        # Training the discriminator
        optim2.zero_grad()
        disc_outputs = disc(gen_outputs)
        loss = criterion(disc_outputs, target_false)
        
        disc_outputs = disc(inputs)
        loss = loss + criterion(disc_outputs, target_true)
        
        loss.backward()
        optim2.step()
        
        # Training the generator
        optim1.zero_grad()
        gen_outputs = gen(torch.randn([len(inputs), 300], dtype=torch.float32, device=device))
        disc_outputs = disc(gen_outputs)
        loss = criterion(disc_outputs, target_true)
        loss.backward()
        optim1.step()
        
    print(f'epoch {i}')
    
for i in range(5):
    img = gen(torch.randn([1, 300], dtype=torch.float32, device=device))
    img = img.squeeze(0).permute(1,2,0).detach().to(cpu).numpy()
    img = (img * 255.0).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'images/generated/img{i}.png')

# Saving the models
torch.save(gen.state_dict, 'anime_faces_generator.pth')
torch.save(disc.state_dict, 'anime_faces_discriminator.pth')
