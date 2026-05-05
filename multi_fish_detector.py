import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import os

data_root = "/home/nmoran/Downloads"

num_classes = 17

def build_model(num_classes, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, weights


model, weights = build_model(num_classes, pretrained=True)

transform = weights.transforms() if weights is not None else None

train_dir = f"{data_root}/multi_train"
test_dir = f"{data_root}/multi_train"
val_dir   = f"{data_root}/multi_val"

train_dataset = datasets.ImageFolder(root=train_dir, transform=weights.transforms())
test_dataset = datasets.ImageFolder(root=test_dir, transform=weights.transforms())
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=weights.transforms())

class_names = train_dataset.classes
batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
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

train_loss = []

epochs = 20

for epoch in range(epochs):
    print(f'Epoch {epoch} training')
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    epoch_loss /= len(train_loader)
    train_loss.append(epoch_loss)
    correct = 0
    total = 0
    print(f'Epoch {epoch} evaluating')
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, digits=5, target_names=class_names, zero_division=0))



all_preds = []
all_labels = []
correct = 0
total = 0
print(f'Running Validation')
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

cm = confusion_matrix(all_labels, all_preds)
print(classification_report(all_labels, all_preds, digits=5, target_names=class_names, zero_division=0))

# Binarize labels
y_bin = label_binarize(all_labels, classes=list(range(num_classes)))

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Epoch {epoch})")
plt.legend()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
file = os.path.join('./results', f'res18fish_weights_roc_{timestamp}.png')
plt.savefig(file)

plt.figure()
plt.plot(train_loss)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
file = os.path.join('./results/loss', f'res18fish_weights_loss_{timestamp}.png')
plt.savefig(file)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
disp.plot()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
file = os.path.join('./results/confusion', f'res18fish_weights_cm_{timestamp}.png')
plt.savefig(file)


plt.close()
