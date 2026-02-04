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


train_path = "./data/PlantVillage/train/"
val_path = './data/PlantVillage/val/'

train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
test_dataset   = datasets.ImageFolder(val_path, transform=test_transforms)


val_size = int(len(test_dataset) / 2)
test_size = len(test_dataset) - val_size
test_set, validation_set = torch.utils.data.random_split(
    test_dataset,
    [test_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

batch_size = 64

train_loader = DataLoader(
      train_dataset, 
      batch_size=batch_size, 
      shuffle=True, 
      num_workers=2, 
      generator=torch.Generator().manual_seed(SEED)
)

validation_loader = DataLoader(
      validation_set, 
      batch_size=batch_size, 
      shuffle=False, 
      num_workers=2
)

test_loader = DataLoader(
      test_set, 
      batch_size=batch_size, 
      shuffle=False, 
      num_workers=2
)