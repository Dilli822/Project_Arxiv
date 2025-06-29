import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mobileone import MobileOne  # Ensure your mobileone.py is in same folder

class MobileOneBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # Extract individual stages for FPN
        self.stage0 = base_model.stage0
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = base_model.stage4
        
    def forward(self, x):
        # We need to return intermediate features for FPN
        x = self.stage0(x)
        
        x = self.stage1(x)
        stage1_out = x
        
        x = self.stage2(x)
        stage2_out = x
        
        x = self.stage3(x)
        stage3_out = x
        
        x = self.stage4(x)
        stage4_out = x
        
        return {
            'stage1': stage1_out,
            'stage2': stage2_out, 
            'stage3': stage3_out,
            'stage4': stage4_out
        }

def inspect_model_channels():
    """Helper function to inspect actual channel dimensions"""
    print("Inspecting MobileOne model channels...")
    base_model = MobileOne(width_multipliers=[3.0, 3.0, 3.0, 3.0])
    backbone = MobileOneBackbone(base_model)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        features = backbone(dummy_input)
    
    print("Actual channel dimensions:")
    channels = []
    for stage_name, feature in features.items():
        print(f"{stage_name}: {feature.shape}")
        channels.append(feature.shape[1])
    
    return channels

def load_pretrained_faster_rcnn():
    """Load a pretrained Faster R-CNN model for comparison"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    
    # Load pretrained COCO model (91 classes including background)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def main():
    print("=" * 60)
    print("IMPORTANT: Your MobileOne + Faster R-CNN model is UNTRAINED!")
    print("It will produce random/meaningless detections.")
    print("=" * 60)
    
    choice = input("Choose option:\n1. Test MobileOne backbone (untrained - will show random results)\n2. Use pretrained ResNet50 Faster R-CNN for comparison\nEnter choice (1 or 2): ")
    
    if choice == "2":
        print("Loading pretrained Faster R-CNN with ResNet50 backbone...")
        model = load_pretrained_faster_rcnn()
        model_name = "Pretrained Faster R-CNN (ResNet50)"
    else:
        print("Creating MobileOne + Faster R-CNN (UNTRAINED)...")
        # First, inspect the actual channel dimensions
        actual_channels = inspect_model_channels()
        
        # Load MobileOne-S4 with width multipliers
        base_model = MobileOne(width_multipliers=[3.0, 3.0, 3.0, 3.0])
        backbone = MobileOneBackbone(base_model)
        
        # Create FPN on top of backbone with correct channel dimensions
        print(f"Using channels: {actual_channels}")
        
        backbone_fpn = BackboneWithFPN(
            backbone,
            return_layers={
                'stage1': '0',  # 1/4 resolution
                'stage2': '1',  # 1/8 resolution  
                'stage3': '2',  # 1/16 resolution
                'stage4': '3'   # 1/32 resolution
            },
            in_channels_list=actual_channels,  # Use actual measured channels
            out_channels=256,
        )
        
        # Create Faster R-CNN model (91 classes like COCO for fair comparison)
        model = FasterRCNN(backbone=backbone_fpn, num_classes=91)
        model.eval()
        model_name = "MobileOne + Faster R-CNN (UNTRAINED)"
    
    # Load image
    img_path = "skin_cancer.jpeg"
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0)
    
    print("Running inference...")
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # IMPORTANT: Use much higher confidence threshold to filter out noise
    confidence_threshold = 0.8 if choice == "2" else 0.95  # Higher threshold for untrained model
    
    print(f"Total raw detections: {len(outputs[0]['boxes'])}")
    print(f"Using confidence threshold: {confidence_threshold}")
    
    # Draw boxes
    img_cv = cv2.imread(img_path)
    detection_count = 0
    
    # COCO class names for pretrained model
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    for i, box in enumerate(outputs[0]['boxes']):
        score = outputs[0]['scores'][i].item()
        if score > confidence_threshold:
            detection_count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add class label and confidence score
            if choice == "2" and len(outputs[0]['labels']) > i:
                class_id = outputs[0]['labels'][i].item()
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                label = f"{class_name}: {score:.2f}"
            else:
                label = f"obj: {score:.2f}"
                
            cv2.putText(img_cv, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Show image with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'{model_name}\n{detection_count} detections (confidence > {confidence_threshold})')
    plt.show()
    
    # Print detection results
    print(f"\nFound {detection_count} confident detections (score > {confidence_threshold})")
    if detection_count == 0:
        print("No confident detections found.")
        if choice == "1":
            print("This is expected for an untrained model!")
            print("To get meaningful results, you need to:")
            print("1. Train the model on a dataset (like COCO)")
            print("2. Or load pretrained weights")
            print("3. Or use option 2 to see pretrained model results")
    
    for i, (box, score) in enumerate(zip(outputs[0]['boxes'], outputs[0]['scores'])):
        if score > confidence_threshold:
            if choice == "2" and len(outputs[0]['labels']) > i:
                class_id = outputs[0]['labels'][i].item()
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                print(f"Detection {i+1}: {class_name}, Box {box.tolist()}, Score: {score:.3f}")
            else:
                print(f"Detection {i+1}: Box {box.tolist()}, Score: {score:.3f}")

if __name__ == "__main__":
    main()