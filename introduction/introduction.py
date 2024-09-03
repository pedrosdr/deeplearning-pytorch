import torch
import torch.nn as nn
import torch.nn.functional as f

x = torch.rand(5, 3)
print(x)

x = torch.zeros(1, 3, dtype=torch.float32)
print(x)

x = torch.ones(20, 10, dtype=torch.int32)
print(x)

x = torch.rand(10, 3)
print(x)

twos = torch.ones(2, 3, dtype=torch.float16) * 2
print(twos)

torch.manual_seed(1)
x = torch.rand(3, 3, dtype=torch.float32)
print(torch.det(x))
print(torch.std(x))
print(torch.std_mean(x))
print(torch.inverse(x)@x)

# Matrix multiplication
torch.manual_seed(1)
x = torch.rand(3, 2)
wx = torch.rand(2, 4)
y = torch.rand(3, 2)
wy = torch.rand(2, 4)

z = x @ wx + y @ wy
print(x)


torch.manual_seed(1)
x = torch.rand(3, 2)
wx = torch.rand(2, 4)
y = torch.rand(3, 2)
wy = torch.rand(2, 4)

x_y = torch.concat([x,y], 1)
wx_wy = torch.concat([wx, wy], 0)
z = x_y @ wx_wy
print(x)



# Differentiation Engine
x = torch.randn(1,10, requires_grad=True)
prev_h = torch.randn(1, 20, requires_grad=True)
w_h = torch.randn(20, 20, requires_grad=True)
w_x = torch.randn(20, 10, requires_grad=True)

i2h = torch.mm(w_x, x.t())
h2h = torch.mm(w_h, prev_h.t())

next_h = i2h + h2h
next_h = next_h.tanh()

loss = next_h.sum()
loss.backward()

# Building models
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = f.max_pool2d(x, (2,2))
        
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, (2,2))
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.fc1(x)
        x = f.relu(x)
        
        x = self.fc2(x)
        x = f.relu(x)
        
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features

net = LeNet()
input = torch.rand(1,1,32,32)           
output = net(input)
print(output)

