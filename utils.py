import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings
import os
import contextlib

class DINOv2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Load Frozen DINOv2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore") # We suppress stdout to keep logs clean
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        
        # Projector: Maps DINO embedding (384) to YOLO channels (c2), We define this in init so weights are trained!
        self.projector = nn.Sequential(
            nn.Conv2d(384, c2, kernel_size=1),
            nn.Upsample(scale_factor=1, mode='bilinear') 
        )

    def preprocess(self, tensor, imgsz=512, patch_size=14):
        # Resize to be divisible by patch_size (14)
        imgsz = round(imgsz / patch_size) * patch_size
        transform = T.Compose([
            T.Resize((imgsz, imgsz), antialias=True), 
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(tensor)

    def forward(self, input_tensor):
        # 1. Extract Global Embedding
        with torch.no_grad():
            processed = self.preprocess(input_tensor)
            embedding = self.model(processed) # Shape: (Batch, 384)

        # 2. Broadcast to Spatial Dimensions, (Batch, 384) -> (Batch, 384, 1, 1)
        embedding = embedding.unsqueeze(-1).unsqueeze(-1)
        
        # 3. Align with YOLO Size (Stride 32)
        h, w = input_tensor.shape[2] // 32, input_tensor.shape[3] // 32
        
        # 4. Project and Upsample, Resize the 1x1 embedding to HxW feature map
        feature_map = nn.functional.interpolate(
            self.projector[0](embedding), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        
        return feature_map

class ConvDummy(nn.Module):
    # Accepts whatever args the parser sends, but ignores them
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        return x
