import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

cuda = torch.device('cuda')
cpu = torch.device('cpu')

x = datasets.MNIST(root='.',  train=False, download=True)
x = np.array([np.array(image) for image, _ in x])
x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
x = torch.tensor(x, dtype=torch.float32)
x = x / 255.0
x = x.view(-1, 28*28)

dataset = torch.utils.data.TensorDataset(x)
dataloader = torch.utils.data.DataLoader(x, batch_size=50)

# decoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense1 = nn.Linear(784, 50)
        
    def forward(self, x):
        x = f.relu(self.dense1(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense1 = nn.Linear(50, 784)
        
    def forward(self, x):
        x = f.sigmoid(self.dense1(x))
        return x
    
    
encoder = Encoder().to(cuda)
decoder = Decoder().to(cuda)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=0.001
)
criterion = nn.MSELoss()

for epoch in range(200):
    losses = []
    for inputs in dataloader:
        inputs = inputs.to(cuda)
        optimizer.zero_grad()
        outputs = decoder(encoder(inputs))
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    losses = np.array(losses)
    print(f'epoch: {epoch}, loss: {losses.mean():.4f}')
    
image = x[4,:].detach().view(28, 28, 1).numpy()
output = decoder(encoder(x[4,:].to(cuda).view(1,784)))
output = output.to(cpu).detach().view(28, 28, 1).numpy()

plt.imshow(image)
plt.imshow(output)
