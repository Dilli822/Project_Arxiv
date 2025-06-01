import torch
import torch.nn as nn
from mobileone import mobileone
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# History dictionary
history = {
    "epoch": [],
    "phase": [],
    "loss": [],
    "accuracy": []
}

# Hyperparameters
batch_size = 64
initial_epochs = 10
finetune_epochs = 20
initial_lr = 0.001
finetune_lr = 0.0001

# Data transformations
train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder('/Users/dilli/Desktop/mobilenetv4.pytorch/skin_dataset_resized/train_set', transform=transform)
val_dataset = datasets.ImageFolder('/Users/dilli/Desktop/mobilenetv4.pytorch/skin_dataset_resized/val_set', transform=transform)

# Compute weighted sampler for imbalanced dataset
labels = [label for _, label in train_dataset]
class_count = torch.bincount(torch.tensor(labels))
class_weights = 1. / class_count.float()
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load MobileOne pretrained model
model = mobileone(variant='s4')
checkpoint = torch.load('/Users/dilli/Desktop/mobilenetv4.pytorch/mobileone_s4_unfused.pth.tar', map_location=device)
model.load_state_dict(checkpoint)

# Replace last layer
num_features = model.linear.in_features
model.linear = nn.Linear(num_features, 2)
model = model.to(device)

# Freeze all except final layer initially
for param in model.parameters():
    param.requires_grad = False
for param in model.linear.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

# Track best validation accuracy for saving best model
best_val_acc = 0.0
best_model_path = "best_mobileone_model.pth"

# Initial training
for epoch in range(initial_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{initial_epochs} Training", leave=False)
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)
        train_bar.set_postfix(loss=f"{running_loss/total_samples:.4f}", acc=f"{running_corrects/total_samples:.4f}")

    print(f"Epoch {epoch+1} Train Loss: {running_loss/total_samples:.4f}, Train Acc: {running_corrects/total_samples:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{initial_epochs} Validation", leave=False)
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += (preds == labels).sum().item()
            val_samples += labels.size(0)
            val_bar.set_postfix(acc=f"{val_corrects/val_samples:.4f}")

    current_val_acc = val_corrects / val_samples
    print(f"Epoch {epoch+1} Val Loss: {val_loss/val_samples:.4f}, Val Acc: {current_val_acc:.4f}")

    # Save best model
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved Best Model with Val Acc: {best_val_acc:.4f}")

    # Log training and validation metrics
    history["epoch"].append(epoch + 1)
    history["phase"].append("train")
    history["loss"].append(running_loss / total_samples)
    history["accuracy"].append(running_corrects / total_samples)
    history["epoch"].append(epoch + 1)
    history["phase"].append("val")
    history["loss"].append(val_loss / val_samples)
    history["accuracy"].append(current_val_acc)

# Fine-tune entire model
print("Unfreezing all layers for fine-tuning...")
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

for epoch in range(finetune_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    train_bar = tqdm(train_loader, desc=f"FineTune Epoch {epoch+1}/{finetune_epochs}", leave=False)
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)
        train_bar.set_postfix(loss=f"{running_loss/total_samples:.4f}", acc=f"{running_corrects/total_samples:.4f}")

    print(f"FineTune Epoch {epoch+1} Train Loss: {running_loss/total_samples:.4f}, Train Acc: {running_corrects/total_samples:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"FineTune Epoch {epoch+1}/{finetune_epochs} Validation", leave=False)
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += (preds == labels).sum().item()
            val_samples += labels.size(0)
            val_bar.set_postfix(acc=f"{val_corrects/val_samples:.4f}")

    current_val_acc = val_corrects / val_samples
    print(f"FineTune Epoch {epoch+1} Val Loss: {val_loss/val_samples:.4f}, Val Acc: {current_val_acc:.4f}")

    # Save best model
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved Best Model with Val Acc: {best_val_acc:.4f}")

    # Log fine-tuning training and validation metrics
    history["epoch"].append(initial_epochs + epoch + 1)
    history["phase"].append("train")
    history["loss"].append(running_loss / total_samples)
    history["accuracy"].append(running_corrects / total_samples)
    history["epoch"].append(initial_epochs + epoch + 1)
    history["phase"].append("val")
    history["loss"].append(val_loss / val_samples)
    history["accuracy"].append(current_val_acc)

# Save training history log
print(train_dataset.class_to_idx)
df = pd.DataFrame(history)
df.to_csv("training_log.csv", index=False)
print("Training log saved to training_log.csv")

# Plot loss and accuracy curves
epochs = df[df["phase"] == "train"]["epoch"].values
train_loss = df[df["phase"] == "train"]["loss"].values
val_loss = df[df["phase"] == "val"]["loss"].values
train_acc = df[df["phase"] == "train"]["accuracy"].values
val_acc = df[df["phase"] == "val"]["accuracy"].values

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_loss, label="Val Loss", marker='x')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Val Accuracy", marker='x')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot_using_simple_apporach.png")
plt.show()
print("Plot saved as training_plot_using_simple_apporach.png")

# Optionally: Load best model after training for evaluation or further use
model.load_state_dict(torch.load(best_model_path))
model = model.to(device)
print(f"Best model loaded with validation accuracy: {best_val_acc:.4f}")

# Epoch 1 Train Loss: 0.2833, Train Acc: 0.8884                                                                                                                 
# Epoch 1 Val Loss: 0.2461, Val Acc: 0.9155  
# Epoch 2 Train Loss: 0.2363, Train Acc: 0.9052                                                                                                                 
# Epoch 2 Val Loss: 0.2529, Val Acc: 0.9055                                                                                                                     
# Epoch 3 Train Loss: 0.2048, Train Acc: 0.9195                                                                                                                 
# Epoch 3 Val Loss: 0.2496, Val Acc: 0.9000                                                                                                                     
# Epoch 4 Train Loss: 0.2056, Train Acc: 0.9199                                                                                                                 
# Epoch 4 Val Loss: 0.2550, Val Acc: 0.8991                                                                                                                     
# Epoch 5 Train Loss: 0.2020, Train Acc: 0.9224                                                                                                                 
# Epoch 5 Val Loss: 0.2403, Val Acc: 0.9000                                                                                                                     
# Epoch 6 Train Loss: 0.1886, Train Acc: 0.9261                                                                                                                 
# Epoch 6 Val Loss: 0.2199, Val Acc: 0.9264 
# Epoch 8 Train Loss: 0.1793, Train Acc: 0.9297                                                                                                                 
# Epoch 8 Val Loss: 0.2109, Val Acc: 0.9264 
# Epoch 9 Train Loss: 0.1816, Train Acc: 0.9252                                                                                                                 
# Epoch 9 Val Loss: 0.2047, Val Acc: 0.9318  