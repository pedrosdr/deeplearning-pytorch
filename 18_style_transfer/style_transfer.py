from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim 
import torch.nn.functional as f
from torchvision import transforms, models

torch.manual_seed(71)

device = torch.device("cuda")
cpu = torch.device("cpu")

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
model = model.to(device)
print(model)

new_img = Image.open("landscape.jpeg")
style_img = Image.open("painting.jpg")

transform = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

new_img = transform(new_img)
style_img = transform(style_img)

new_img = new_img.unsqueeze(0)
style_img = style_img.unsqueeze(0)

def plot(img):
    img = img.to(cpu).clone().detach().numpy().squeeze()
    img = img.transpose(1,2,0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    plt.imshow(img)
    
plot(style_img)
plot(new_img)
