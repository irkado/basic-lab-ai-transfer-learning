from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from torch.utils.data import random_split
import numpy as np
import random


SEED = 42
random.seed(SEED) 
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


path = "./Data_Preparation/data/color"

train_data_full = datasets.ImageFolder(path, transform=train_transforms)
test_data_full = datasets.ImageFolder(path, transform=test_transforms)

num_train = len(train_data_full)
indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(SEED)).tolist()

train_size = int(0.7 * num_train)
val_size = int(0.15 * num_train)
test_size = num_train - train_size - val_size

train_indices = indices[:train_size]
val_indices   = indices[train_size : train_size+val_size]
test_indices  = indices[train_size+val_size:]

train_dataset = Subset(train_data_full, train_indices)
val_dataset  = Subset(test_data_full, val_indices)
test_dataset = Subset(test_data_full, test_indices)


batch_size = 64

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2, 
    generator=torch.Generator().manual_seed(SEED)
)

validation_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)