import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from PIL import Image

# Adjust this to where your data actually lives
data_root = "/Users/nmoran/Downloads"

# class ImageDataset(Dataset):
#     def __init__(self, samples, transform=None):
#         self.samples = samples
#         self.transform = transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         image_path = self.samples[idx]
#         label = None
#         if "nofish" in image_path:
#           label = 0
#         else:
#           label = 1
#         image = Image.open(image_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label

def build_model(num_classes, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, weights


# Need to bring in traininger data here, have train and val samples.
# Can load these from Google Drive. If done correctly, should just work with the code below.

num_classes = 2
model, weights = build_model(num_classes, pretrained=True)

transform = weights.transforms() if weights is not None else None

# train_transform = transforms.Compose([
#     transform,
#     transforms.ToTensor(),
# ])

# val_transform = transforms.Compose([
#     transform,
#     transforms.ToTensor(),
# ])

train_dir = f"{data_root}/train"
val_dir   = f"{data_root}/val"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=transforms.ToTensor())

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 100

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1}: val acc = {correct / total:.4f}")
    # Need to save this and maybe make a plot? Or a confusion matrix for the classes for binary case?
    # We could also do multi class for each fish type and see how that compares
