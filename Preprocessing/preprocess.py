import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

torch.manual_seed(42)

# -------- Dummy U-Net Model (for testing only) --------
class UNetStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Conv2d(3, 1, kernel_size=1)  # Output mask with 1 channel

    def forward(self, x):
        return torch.sigmoid(self.out(x))  # Fake binary mask


# --------- Preprocessing Filters ---------
class MedianFilterTransform:
    def __init__(self, ksize=3):
        self.ksize = ksize
    def __call__(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        filtered = cv2.medianBlur(img_cv, self.ksize)
        return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))

class CLAHEFilterTransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        merged = cv2.merge((cl,a,b))
        clahe_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(clahe_rgb)

class HairRemovalTransform:
    def __call__(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17,17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat,15,255,cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(img_cv, mask, 1, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

# --------- U-Net Segmentation + Mask Application ---------
class UNetSegmentationTransform:
    def __init__(self, model, device='cpu', input_size=224):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.input_size = input_size
        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __call__(self, img):
        # Preprocess input image
        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)  # [1,3,H,W]
        with torch.no_grad():
            mask_pred = self.model(input_tensor)  # Output shape: [1,1,H,W]
            mask_pred = mask_pred.squeeze().cpu().numpy()
            mask_bin = (mask_pred > 0.5).astype(np.uint8)  # Binary mask

        # Resize mask back to original image size
        mask_bin = cv2.resize(mask_bin, img.size, interpolation=cv2.INTER_NEAREST)

        # Apply mask on original image (black background outside lesion)
        img_np = np.array(img)
        masked_img = img_np.copy()
        masked_img[mask_bin == 0] = 0  # zero out background pixels

        return Image.fromarray(masked_img)

# --------- Full Pipeline ---------
class FullSkinPipeline:
    def __init__(self, segmentation_model, device='mps'):
        self.median = MedianFilterTransform()
        self.hair = HairRemovalTransform()
        self.clahe = CLAHEFilterTransform()
        self.segment = UNetSegmentationTransform(segmentation_model, device=device)
       
    def __call__(self, img):
        img = self.median(img)
        img = self.hair(img)
        img = self.clahe(img)
        img = self.segment(img)      # Segment and mask lesion
        return img

import os

def process_folder(input_dir, output_dir, pipeline):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert("RGB")

            processed_img = pipeline(img)

            save_path = os.path.join(output_dir, filename)
            processed_img.save(save_path)
            print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    device = 'mps' if torch.mps.is_available() else 'cpu'
    unet_model = UNetStub()
    pipeline = FullSkinPipeline(unet_model, device=device)

    # Set your input/output folders here
    input_folder = "/Users/dilli/Downloads/MergedSC_split/train/malignant"      # change to  input directory
    output_folder = "/Users/dilli/Desktop/Project_Arxiv/Preprocessing/processed_images/train/malignant"    # change to your output directory

    process_folder(input_folder, output_folder, pipeline)
