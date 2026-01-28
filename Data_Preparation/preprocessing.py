from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224,padding=8),
    transforms.RandomRotation(degrees=15),
    transforms.GaussianBlur(5,(0.1,1)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
    ),
])

test_transforms = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SEED = 42

path = "./Data_Preparation/data/color"

image_data = datasets.ImageFo