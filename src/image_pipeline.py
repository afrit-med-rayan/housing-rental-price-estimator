import os
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm
import ast

class ImagePipeline:
    def __init__(self, model_weight="IMAGENET1K_V1"):
        # EfficientNet-B0 pretrained
        self.model = models.efficientnet_b0(weights=model_weight)
        self.model.classifier = torch.nn.Identity()  # remove classifier -> output = 1280 dim
        self.model.eval()

        # Image transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def extract_image_feature(self, img_path):
        """Extracts features from a single image."""
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = self.model(img).squeeze().numpy()
            return feat
        except Exception as e:
            # print(f"Error processing {img_path}: {e}")
            return np.zeros(1280)

    def get_aggregated_features(self, image_list, base_folder=""):
        """Extracts and averages features from a list of images."""
        if not image_list:
            return np.zeros(1280)
        
        # Handle string representation of list if passed
        if isinstance(image_list, str):
            try:
                image_list = ast.literal_eval(image_list)
            except:
                return np.zeros(1280)

        feats = []
        for f in image_list:
            full_path = os.path.join(base_folder, f)
            feats.append(self.extract_image_feature(full_path))
        
        if not feats:
            return np.zeros(1280)
            
        return np.mean(np.vstack(feats), axis=0)

    def batch_process(self, local_images_series, base_folder, desc="Extracting image features"):
        """Processes a series of image lists."""
        image_features = []
        for imgs in tqdm(local_images_series, desc=desc):
            feat = self.get_aggregated_features(imgs, base_folder)
            image_features.append(feat)
        return np.array(image_features)
