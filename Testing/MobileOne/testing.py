import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from mobileone import mobileone  # Make sure your mobileone.py exists
import cv2
from PIL import Image

# Device setup
torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🔋 Using device: {device}")

# Median filter class (not used but defined)
# Median filter class using OpenCV
class MedianFilterTransform:
    def __init__(self, ksize=3):
        self.ksize = ksize

    def __call__(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        filtered = cv2.medianBlur(img_cv, self.ksize)
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

# Laplace Filter Transform
class LaplaceFilterTransform:
    def __init__(self, ksize=3):  # ksize must be odd and positive
        self.ksize = ksize

    def __call__(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        laplace = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=self.ksize)
        laplace = cv2.convertScaleAbs(laplace)  # Convert back to 8-bit

        # Stack back to 3 channels for consistency
        laplace_3ch = cv2.merge([laplace, laplace, laplace])
        laplace_rgb = cv2.cvtColor(laplace_3ch, cv2.COLOR_BGR2RGB)
        return Image.fromarray(laplace_rgb)

# Instantiate transforms
median_filter = MedianFilterTransform(ksize=3)
laplace_filter = LaplaceFilterTransform(ksize=3)

# Paths
MODEL_PATH = os.path.expanduser("/Users/dilli/Desktop/Project_Arxiv/Training/MobileOne/Classification/Training_Models_History/June29/_0.52958_best_model_by_test_loss.pth")
TEST_DIR = os.path.expanduser("/Users/dilli/Downloads/MergedSC_split/test")

median_filter = MedianFilterTransform(ksize=5)
if median_filter is None:
    print("❗ MedianFilterTransform is None, Median Filter is not used.")
else:   
    print("✅ MedianFilterTransform is ready to use.")
    
# Transform
transform = transforms.Compose([
    median_filter,
    laplace_filter,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# DataLoader
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ Create model with SAME classifier as training
model = mobileone(num_classes=2, variant='s4')
in_features = model.linear.in_features
model.linear = torch.nn.Sequential(
    torch.nn.Dropout(p=0),
    torch.nn.Linear(in_features, 2)
)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("🌀 Model loaded and ready...")

# Inference
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4)
print("📋 Classification Report:\n")
print(report)

# Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Overall Accuracy: {accuracy:.4f}")

# ROC Curve
if all_probs.shape[1] == 1:
    probs = all_probs.ravel()
else:
    probs = all_probs[:, 1]  # prob for positive class

fpr, tpr, _ = roc_curve(all_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()
