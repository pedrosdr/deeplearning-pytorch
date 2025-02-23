from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim 
import torch.nn.functional as f
from torchvision import transforms, models
from datetime import datetime

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


def extract_features(image):
    layers = [0, 5, 10, 19, 28]
    
    features = {}
    
    for n in range(len(model.features)):
        image = model.features[n](image)
        if n in layers:
            features[n] = image
            
    return features

new_img = new_img.to(device)
style_img = style_img.to(device)

features_new_img = extract_features(new_img)
features_style_img = extract_features(style_img)


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h*w)
    
    gram = torch.mm(tensor, tensor.t())
    return gram


style_img_grams = {
    layer: gram_matrix(feature) for layer, feature in features_style_img.items()
}

style_weights = {
    0: 1.,
    5: 0.8,
    10: 0.6,
    19: 0.4,
    28: 0.2
}

content_weight = 1
style_weight = 1e6

target_img = new_img.clone().requires_grad_(True).to(device)

optimizer = optim.Adam([target_img], lr=0.002)

# Training
epochs = 3000
print_every = 500

start = datetime.now()
for epoch in range(epochs):
    features_target_img = extract_features(target_img)
    
    # content loss
    target_img_content_features = features_target_img[28]
    new_img_content_features = features_new_img[28]
    
    content_loss = f.mse_loss(target_img_content_features, new_img_content_features)
    print(content_loss.item())
