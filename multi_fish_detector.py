import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def main():

    num_classes = 17

    has_weights = False

    def build_model(num_classes, pretrained=True):
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, weights


    model, weights = build_model(num_classes, pretrained=has_weights)

    device = None
    data_root = None
    workers = None
    if torch.backends.mps.is_available():
        data_root = "/Users/nmoran/Downloads/prepared"
        workers = 10
        device = torch.device("mps")
    elif torch.cuda.is_available():
        data_root = '/home/nmoran/Downloads'
        workers = 16
        device = torch.device("cuda")
    else:
        raise ValueError('Probably dont want to try this on cpu')

    print("Using device:", device)
    model = model.to(device)

    transform = weights.transforms() if weights is not None else ResNet18_Weights.DEFAULT.transforms()

    train_dir = f"{data_root}/multi_train"
    test_dir = f"{data_root}/multi_train"
    val_dir   = f"{data_root}/multi_val"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_dir,   transform=transform)

    targets = torch.tensor(train_dataset.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    class_names = train_dataset.classes
    batch_size = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loss = []

    epochs = 20

    all_probs = []
    all_labels = []

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
        epoch_preds = []
        epoch_labels = []
        epoch_probs = []
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
                epoch_preds.append(preds.cpu())
                epoch_labels.append(labels.cpu())
                epoch_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
        epoch_preds = torch.cat(epoch_preds).numpy()
        epoch_labels = torch.cat(epoch_labels).numpy()
        epoch_probs = torch.cat(epoch_probs).numpy()

        print(confusion_matrix(epoch_labels, epoch_preds))
        print(classification_report(epoch_labels, epoch_preds, digits=5, target_names=class_names, zero_division=0))



    val_preds = []
    val_labels = []
    correct = 0
    total = 0
    print(f'Running Validation')
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_preds.append(preds.cpu())
            val_labels.append(labels.cpu())
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()

    cm = confusion_matrix(val_labels, val_preds)
    report = classification_report(val_labels, val_preds, digits=5, target_names=class_names, zero_division=0, output_dict=True)

# Calculate TP, FP, FN, TN per class
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

# Convert to float for metric calculations
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

# Calculate rates per class
    TPR = TP / (TP + FN)  # Sensitivity/Recall
    TNR = TN / (TN + FP)  # Specificity
    PPV = TP / (TP + FP)  # Precision
    FPR = FP / (FP + TN)  # False Positive Rate

    summary = {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'avg_TPR': TPR.mean(),
        'avg_FPR': FPR.mean()
    }

    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 30)

    print(f"{'FP':<20} {FP}")
    print(f"{'FN':<20} {FN}")
    print(f"{'TP':<20} {TP}")
    print(f"{'TN':<20} {TN}")
    print(f"{'Accuracy':<20} {summary['accuracy']:>10.4f}")
    print(f"{'Macro F1':<20} {summary['macro_f1']:>10.4f}")
    print(f"{'Weighted F1':<20} {summary['weighted_f1']:>10.4f}")
    print(f"{'Avg TPR (Recall)':<20} {summary['avg_TPR']:>10.4f}")
    print(f"{'Avg FPR':<20} {summary['avg_FPR']:>10.4f}")

# Binarize labels
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    y_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    weight_string = None
    if has_weights:
        weight_string = 'weights'
    else:
        weight_string = 'noweights'

# Plot
    plt.figure(figsize=(12,8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Res18")
    plt.legend()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file = os.path.join('./results/roc', f'res18fish_{weight_string}_roc_{timestamp}.png')
    plt.savefig(file, dpi=300)

    plt.figure(figsize=(12,8))
    plt.plot(train_loss)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file = os.path.join('./results/loss', f'res18fish_{weight_string}_loss_{timestamp}.png')
    plt.savefig(file, dpi=300)

    plt.figure(figsize=(20,12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file = os.path.join('./results/confusion', f'res18fish_{weight_string}_cm_{timestamp}.png')
    plt.savefig(file, dpi=300)


    plt.close()

if __name__ == '__main__':
    main()
