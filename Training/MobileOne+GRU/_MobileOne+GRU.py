import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

# Import MobileOne model
try:
    from mobileone import mobileone
    MobileOne = mobileone
except ImportError:
    raise ImportError("Could not import mobileone module. Ensure mobileone.py is in the correct directory.")

# Setup device-agnostic code
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"📱Using device: {device}")

# Define paths
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path(os.path.expanduser('/Users/dilli/Desktop/Project_Arxiv/Models_Saved/mobileone_s4_unfused.pth.tar'))
TRAIN_DIR = Path(os.path.expanduser("~/Downloads/MergedSC_split/train"))
TEST_DIR = Path(os.path.expanduser("~/Downloads/MergedSC_split/test"))
VAL_DIR = Path(os.path.expanduser("~/Downloads/MergedSC_split/val"))

# Validate paths
for path in [CHECKPOINT_PATH, TRAIN_DIR, TEST_DIR, VAL_DIR]:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

# Median filter class using OpenCV
class MedianFilterTransform:
    def __init__(self, ksize=5):  # ksize must be odd and > 1
        self.ksize = ksize

    def __call__(self, img):
        # Convert PIL to OpenCV (BGR)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply OpenCV's built-in median filter
        filtered = cv2.medianBlur(img_cv, self.ksize)
        
        # Convert back to PIL (RGB)
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

median_filter = MedianFilterTransform(ksize=5)

# Data transforms
train_transform = transforms.Compose([
    # Initial resizing
    transforms.Resize(256),
    
    # Spatial transformations
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Random crop with scale variation
    transforms.RandomHorizontalFlip(p=0.5),  # Left-right flip
    transforms.RandomVerticalFlip(p=0.2),  # Top-bottom flip
    transforms.RandomRotation(30),  # Random rotation (±30°)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),  # Shift + shear
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # 3D perspective warping
    
    # Color & texture transformations
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color variations
    transforms.RandomGrayscale(p=0.1),  # Random grayscale
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Sharpness variation
    transforms.RandomAutocontrast(p=0.2),  # Auto-contrast adjustment
    transforms.RandomPosterize(bits=4, p=0.1),  # Reduce color bits
    transforms.RandomSolarize(threshold=192, p=0.1),  # Solarization effect
    
    # Noise & blur
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Random blur
    
    # Tensor conversion & normalization
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Random masking
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=test_transform)

# Get class names and number of classes
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Create data loaders
BATCH_SIZE = 24
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")


# Custom MobileOne + GRU Model
class MobileOneGRUClassifier(nn.Module):
    def __init__(self, num_classes=2, variant='s4', hidden_size=256, num_layers=2, dropout=0.3):
        super(MobileOneGRUClassifier, self).__init__()
        
        # Load pretrained MobileOne as feature extractor
        self.feature_extractor = MobileOne(num_classes=1000, variant=variant)
        
        # Remove the final classification layer to use as feature extractor
        # Get the input size for the GRU (typically 2048 for MobileOne-S4)
        feature_size = self.feature_extractor.linear.in_features
        self.feature_extractor.linear = nn.Identity()  # Remove final layer
        
        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Add GRU layers (simple, not bidirectional)
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # No *2 since not bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features using MobileOne
        features = self.feature_extractor(x)  # Shape: (batch_size, feature_size)
        
        # Reshape for GRU: (batch_size, seq_len=1, feature_size)
        features = features.unsqueeze(1)
        
        # Pass through GRU
        gru_out, _ = self.gru(features)  # Shape: (batch_size, seq_len=1, hidden_size)
        
        # Take the last output (since seq_len=1, this is just the output)
        gru_out = gru_out.squeeze(1)  # Shape: (batch_size, hidden_size)
        
        # Apply dropout and classify
        output = self.dropout(gru_out)
        output = self.classifier(output)
        
        return output

def load_pretrained_mobileone(model, checkpoint_path, device):
    """Load pretrained MobileOne weights into the feature extractor"""
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading pretrained MobileOne weights from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

            # Load weights into feature extractor (excluding the final linear layer)
            model_dict = model.feature_extractor.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and not k.startswith('linear')}
            model_dict.update(pretrained_dict)
            model.feature_extractor.load_state_dict(model_dict)
            print("Pretrained MobileOne weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using random initialization.")
    else:
        print("No checkpoint found. Using random initialization.")

# Create model
model = MobileOneGRUClassifier(num_classes=num_classes, variant='s4', hidden_size=256, num_layers=2, dropout=0.3)

# Load pretrained weights
load_pretrained_mobileone(model, CHECKPOINT_PATH, device)

# Move model to device
model = model.to(device)

# Print model info
print(f"Model created with {num_classes} classes")
print(f"Feature extractor frozen: {not next(model.feature_extractor.parameters()).requires_grad}")
print(f"GRU trainable parameters: {sum(p.numel() for p in model.gru.parameters() if p.requires_grad):,}")
print(f"Classifier trainable parameters: {sum(p.numel() for p in model.classifier.parameters() if p.requires_grad):,}")

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer (Adam, only trainable params)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    weight_decay=1e-4
)

# Learning rate scheduler (Reduce LR on Plateau, monitoring val loss)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # we're minimizing val_loss
    factor=0.1,          # reduce LR by 10x
    patience=3,          # wait 3 epochs without improvement
)

def train_step(model, dataloader, loss_fn, optimizer, device):
    """Train the model for one epoch with tqdm progress bar"""
    model.train()
    train_loss, train_acc = 0, 0
    loop = tqdm(dataloader, leave=False, desc='Training')
    
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        train_acc += acc
        
        loop.set_postfix(loss=loss.item(), accuracy=acc)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    """Evaluate the model with tqdm progress bar"""
    model.eval()
    test_loss, test_acc = 0, 0
    loop = tqdm(dataloader, leave=False, desc='Testing')
    
    with torch.inference_mode():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            acc = (test_pred_labels == y).sum().item() / len(test_pred_labels)
            test_acc += acc
            
            loop.set_postfix(loss=loss.item(), accuracy=acc)
            
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def validate_model(model, dataloader, loss_fn, device):
    """Run validation on the provided dataloader and return loss and accuracy"""
    model.eval()
    val_loss = 0
    val_acc = 0
    loop = tqdm(dataloader, leave=False, desc="Validating")
    
    with torch.inference_mode():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            val_pred_labels = val_pred_logits.argmax(dim=1)
            acc = (val_pred_labels == y).sum().item() / len(val_pred_labels)
            val_acc += acc
            
            loop.set_postfix(loss=loss.item(), accuracy=acc)
    
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc

def train_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    best_test_loss = float('inf')
    best_val_loss = float('inf')
    best_test_acc = 0.0
    best_val_acc = 0.0
    best_model_wts = None
    best_acc_model_wts = None

    train_loss_values, train_acc_values = [], []
    test_loss_values, test_acc_values = [], []
    val_loss_values, val_acc_values = [], []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        val_loss, val_acc = validate_model(model, val_dataloader, loss_fn, device)
        
        # scheduler.step()
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}")

        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)
        test_loss_values.append(test_loss)
        test_acc_values.append(test_acc)
        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        # Save best model by test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, MODEL_PATH / "best_mobileone_gru_by_loss.pth")
            print(f"Saved best model by loss at epoch {epoch+1} with test loss {best_test_loss:.5f}")
        
        # Save best model by test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_acc_model_wts = model.state_dict()
            torch.save(best_acc_model_wts, MODEL_PATH / "best_mobileone_gru_by_acc.pth")
            print(f"Saved best model by accuracy at epoch {epoch+1} with test acc {best_test_acc:.5f}")

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_model_wts = model.state_dict()
            torch.save(best_val_model_wts, MODEL_PATH / "best_mobileone_gru_by_val_acc.pth")
            print(f"Saved best model by validation accuracy at epoch {epoch+1} with val acc {best_val_acc:.5f}")

    return {
        "train_loss": train_loss_values,
        "train_acc": train_acc_values,
        "test_loss": test_loss_values,
        "test_acc": test_acc_values,
        "val_loss": val_loss_values,
        "val_acc": val_acc_values
    }

# Set random seeds for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# Start training
print("Starting training with MobileOne + GRU architecture...")
NUM_EPOCHS = 20 # Increased epochs for better training

results = train_model(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     epochs=NUM_EPOCHS,
                     device=device)

# Plot training curves
def plot_loss_curves(results):
    """Plot training and test loss/accuracy curves"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss - MobileOne + GRU")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy - MobileOne + GRU")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(MODEL_PATH / "mobileone_gru_training_curves.png")
    plt.show()

plot_loss_curves(results)

# Save the trained model
MODEL_NAME = "mobileone_gru_trained_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

try:
    with open(MODEL_SAVE_PATH, 'wb') as f:
        torch.save(model.state_dict(), f)
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# Load and test the saved model
def load_model(model_path, num_classes, variant='s4'):
    """Load a saved MobileOne + GRU model"""
    loaded_model = MobileOneGRUClassifier(num_classes=num_classes, variant=variant)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    return loaded_model

# Test loading
loaded_model = load_model(MODEL_SAVE_PATH, num_classes, variant='s4')
loaded_model = loaded_model.to(device)

# Evaluate on validation set
val_loss, val_acc = validate_model(loaded_model, val_dataloader, loss_fn, device)
print(f"📊 Final Validation Loss: {val_loss:.5f} | Final Validation Accuracy: {val_acc:.5f}")

# Make predictions on a sample
def make_predictions(model, dataloader, device):
    """Make predictions and return results"""
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred_probs = torch.softmax(y_pred, dim=1)
            y_pred_class = torch.argmax(y_pred_probs, dim=1)
            
            predictions.extend(y_pred_class.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probabilities.extend(y_pred_probs.cpu().numpy())
    
    return np.array(predictions), np.array(labels), np.array(probabilities)

# Get predictions on test set
test_predictions, test_labels, test_probabilities = make_predictions(loaded_model, test_dataloader, device)

print(f"Made {len(test_predictions)} predictions")
print(f"Test accuracy: {(test_predictions == test_labels).mean():.4f}")

# Print class-wise performance
if num_classes == 2:
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))

print(f"\n🎯 Architecture: MobileOne (frozen) → GRU → Classification Head")
print(f"🔧 Optimizer: Adam (lr=0.001)")
print(f"📊 Final Results: Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}")