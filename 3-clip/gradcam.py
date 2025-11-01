import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from loguru import logger
import glob
import warnings
warnings.filterwarnings('ignore')

class CLIPWrapper(torch.nn.Module):
    """GradCAM"""
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features
        
    def forward(self, x):
        image_features = self.clip_model.get_image_features(pixel_values=x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ self.text_features.T) * self.clip_model.logit_scale.exp()
        return logits

class CLIPGradCAMVisualization:
    def __init__(self, model_path, device='cuda'):
        """
        CLIP Grad-CAM
        Args:
            model_path: path
            device: device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"use: {self.device}")
        
        
        self.load_model(model_path)
        
        
        self.text_labels = [
            "No osteoarthritis, No radiographic features of osteoarthritis.",
            "Doubtful osteoarthritis, Doubtful narrowing of joint space and possible osteophytic lipping.",
            "Mild osteoarthritis, Definite osteophytes and possible narrowing of joint space.",
            "Moderate osteoarthritis, Multiple osteophytes, definite narrowing of joint space, some sclerosis, and possible deformity of bone ends.",
            "Severe osteoarthritis, Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
        ]
        
        self.class_names = [
            "No OA (Grade 0)",
            "Doubtful OA (Grade 1)",
            "Mild OA (Grade 2)",
            "Moderate OA (Grade 3)",
            "Severe OA (Grade 4)"
        ]
        
        
        self._precompute_text_features()
        
        # last transformer block）
        self.target_layer = self.model.vision_model.encoder.layers[-1].layer_norm1
        
    def load_model(self, model_path):
        """load"""
        logger.info(f"from {model_path} load...")
        try:
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("done!")
        except Exception as e:
            logger.error(f"fail: {e}")
            raise e
    
    def _precompute_text_features(self):
        """caculate"""
        logger.info("caculate...")
        
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.text_labels,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        self.text_features = text_features
        logger.info(f"text: {self.text_features.shape}")
    
    def reshape_transform(self, tensor, height=24, width=24):
        """
        transformer Grad-CAM
        CLIP ViT-L/14@336px patch 24x24
        """
        
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        
        
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    def get_prediction(self, image_path):
        """result"""
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            
            logits = (image_features @ self.text_features.T) * self.model.logit_scale.exp()
            probs = F.softmax(logits, dim=-1)
            
            predicted_class = logits.argmax(dim=-1).item()
            confidence = probs[0, predicted_class].item()
        
        return predicted_class, confidence, probs[0].cpu().numpy()
    
    def visualize_gradcam(self, image_path, target_class=None, save_path=None):
        """
        Grad-CAM
        Args:
            image_path: path
            target_class: class
            save_path: path
        """
        
        original_image = Image.open(image_path).convert('RGB')
        rgb_img = np.array(original_image)
        rgb_img = cv2.resize(rgb_img, (336, 336))
        rgb_img_normalized = rgb_img.astype(np.float32) / 255.0
        
        
        pred_class, confidence, probs = self.get_prediction(image_path)
        
        if target_class is None:
            target_class = pred_class
        
        logger.info(f"predict: {self.class_names[pred_class]} (confidence: {confidence:.2%})")
        logger.info(f"target: {self.class_names[target_class]}")
        
        
        wrapped_model = CLIPWrapper(self.model, self.text_features)
        wrapped_model.to(self.device)
        wrapped_model.eval()
        
        
        cam = GradCAM(
            model=wrapped_model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        )
        
        
        inputs = self.processor(images=original_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        
        targets = [ClassifierOutputTarget(target_class)]
        
        
        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        
        cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
        
        
        visualization = show_cam_on_image(rgb_img_normalized, cam_resized, use_rgb=True)
        
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title('(a) Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        
        im = axes[0, 1].imshow(cam_resized, cmap='jet')
        axes[0, 1].set_title(f'(b) Grad-CAM Heatmap\nTarget: {self.class_names[target_class]}', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        
        axes[0, 2].imshow(visualization)
        axes[0, 2].set_title('(c) Grad-CAM Overlay', fontsize=14)
        axes[0, 2].axis('off')
        
        
        y_pos = np.arange(len(self.class_names))
        bars = axes[1, 0].barh(y_pos, probs)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(self.class_names)
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_title('(d) Prediction Probabilities', fontsize=14)
        axes[1, 0].set_xlim(0, 1)
        

        bars[pred_class].set_color('red')
        bars[target_class].set_edgecolor('blue')
        bars[target_class].set_linewidth(3)
        
        
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            width = bar.get_width()
            axes[1, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1%}', ha='left', va='center')
        
        
        threshold = np.percentile(cam_resized, 80)  # 20%
        high_activation_mask = cam_resized > threshold
        masked_image = rgb_img.copy()
        masked_image[~high_activation_mask] = masked_image[~high_activation_mask] * 0.3
        axes[1, 1].imshow(masked_image)
        axes[1, 1].set_title('(e) High Activation Regions (Top 20%)', fontsize=14,y=1.05)
        axes[1, 1].axis('off')
        
        
        x = np.arange(0, cam_resized.shape[1])
        y = np.arange(0, cam_resized.shape[0])
        X, Y = np.meshgrid(x, y)
        
        ax3d = fig.add_subplot(2, 3, 6, projection='3d')
        surf = ax3d.plot_surface(X, Y, cam_resized, cmap='jet', alpha=0.8)
        ax3d.set_title('(f) 3D Activation Surface', fontsize=14,y=1.12)
        ax3d.set_xlabel('Width')
        ax3d.set_ylabel('Height')
        ax3d.set_zlabel('Activation')
        
        plt.tight_layout()
        
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"save: {save_path}")
        
        plt.show()
        
        return visualization, cam_resized, probs
    
    def batch_visualize(self, image_dir, output_dir, max_images=None):
        """image"""
        os.makedirs(output_dir, exist_ok=True)
        
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
       
        if max_images is not None:
            image_files = image_files[:max_images]
        
        logger.info(f"find {len(image_files)} images")
        
        for i, img_path in enumerate(image_files, 1):
            img_name = os.path.basename(img_path)
            save_path = os.path.join(output_dir, f"gradcam_{img_name}")
            
            try:
                logger.info(f"process [{i}/{len(image_files)}]: {img_name}")
                self.visualize_gradcam(img_path, save_path=save_path)
            except Exception as e:
                logger.error(f" {img_name} wrong: {e}")
                continue

def main():
    
    model_path = ""
    input_dir = ""  
    output_dir = ""  
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    visualizer = CLIPGradCAMVisualization(model_path)
    
    
    visualizer.batch_visualize(
        image_dir=input_dir,
        output_dir=output_dir,
        max_images=None  
    )
    
    logger.info("ok!")

if __name__ == "__main__":
    main()