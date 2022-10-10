import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import sys
import matplotlib.pyplot as plt
plt.ion()

sys.path.append('../')
from dataloader import ImageLoader, print_batch

# HELPER FUNCTIONS ------------------------------------------------------------------------------------------------------------------------------------------

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



# MAIN LOGIC ------------------------------------------------------------------------------------------------------------------------------------------

data_loader = data.DataLoader(
    ImageLoader(
        '../../val/',
        transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        data_path='../../val',
        partition='val',
        clean_ingredients='../../recipe1M_layers/cleaned_ingredients.json',
        clean_layers='../../recipe1M_layers/cleaned_layers.json'),
    batch_size=1,
    shuffle=False
)

model_conv = models.resnet50(pretrained=True)
model_conv = torch.nn.Sequential(*list(model_conv.children())[:-1])
model_conv = model_conv.to(device)
for params in model_conv.parameters():
    params.requires_grad = False


for i, (input, target) in enumerate(data_loader):
    images = torch.tensor(input[0])
    images = images.to(device)

    output = torch.squeeze(model_conv(images))

    # do t-SNE from here
