import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')
cpu = torch.device('cpu')

base = datasets.CIFAR10(root='.', train=False, transform=transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(base, batch_size=200)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 6, [3,3], padding='same')
        self.bnorm1 = nn.BatchNorm2d(6)
        
        self.dense1 = nn.Linear(1536, 768)
        
    def forward(self, x):
        x = f.max_pool2d(self.bnorm1(f.relu(self.conv1(x))), [2,2])
        x = x.view(-1, 1536)
        x = f.dropout(f.relu(self.dense1(x)), 0.2)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense1 = nn.Linear(768, 1536)
        
        self.upsample1 = nn.UpsamplingBilinear2d([32,32])
        self.conv1 = nn.Conv2d(6, 3, [3,3], padding='same')
        
    def forward(self, x):
        x = f.dropout(f.relu(self.dense1(x)))
        x = x.view(-1, 6, 16, 16)
        x = self.upsample1(x)
        x = f.sigmoid(self.conv1(self.upsample1(x)))
        return x


encoder = Encoder().to(device)
decoder = Decoder().to(device)

opt1 = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=0.0001)
opt2 = torch.optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.0001)

criterion = nn.MSELoss()

for i in range(100):
    losses = []
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        
        opt1.zero_grad()
        opt2.zero_grad()
        
        outputs = decoder(encoder(inputs))
        
        loss = criterion(outputs, inputs)
        loss.backward()
        
        opt1.step()
        opt2.step()
        losses.append(loss.item())
    losses = np.array(losses)
    print(f'epoch {i}, loss: {losses.mean():.3f}')

image = next(iter(dataloader))[0][26:27].to(device)
output = decoder(encoder(image))

image = image.squeeze(0).permute(1,2,0).detach().to(cpu).numpy()
output = output.squeeze(0).permute(1,2,0).detach().to(cpu).numpy()
plt.imshow(output)
plt.imshow(image)
