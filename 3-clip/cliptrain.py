import os
##wandb
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = ""
os.environ["WANDB_CACHE_DIR"] = ""
os.environ["WANDB_CONFIG_DIR"] = ""

import wandb
import random
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
import torch.optim as optim
from torch.optim import lr_scheduler
from loguru import logger
import argparse
from tqdm import tqdm
import json
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class OsteoarthritisDataset(Dataset):
    
    def __init__(self, data_root, processor, is_train=True):
        """
        OAdataset
        Args:
            data_root: train or val
            processor: CLIP334
            is_train: train set
        """
        self.processor = processor
        self.is_train = is_train
        
        
        self.text_labels = [
            "No radiographic features of osteoarthritis.",
            "Doubtful narrowing of joint space and possible osteophytic lipping.",
            "Mild osteoarthritis, Definite osteophytes and possible narrowing of joint space.",
            "Moderate osteoarthritis, Multiple osteophytes, definite narrowing of joint space, some sclerosis, and possible deformity of bone ends.",
            "Severe osteoarthritis, Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
        ]
        
        
        self.samples = []
        self.labels = []
        
        for class_id in range(5):  
            class_dir = os.path.join(data_root, str(class_id))
            if not os.path.exists(class_dir):
                logger.warning(f"not exist: {class_dir}")
                continue
                
            
            image_files = self._get_image_files(class_dir)
            
            if len(image_files) == 0:
                logger.warning(f"no {class_dir} ")
                continue
                
            
        
            for img_path in image_files:
                self.samples.append(img_path)
                self.labels.append(class_id)
            
            logger.info(f"class {class_id}: load {len(image_files)} images")
        
        logger.info(f"finish: {len(self.samples)} samples")
        self._print_class_distribution()
    
    def _get_image_files(self, directory):
        """get images"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, extension)))
            image_files.extend(glob.glob(os.path.join(directory, extension.upper())))
        
        return sorted(image_files)
    
    def _print_class_distribution(self):
        """print"""
        class_counts = [0] * 5
        for label in self.labels:
            class_counts[label] += 1
        
        logger.info("class:")
        for i, count in enumerate(class_counts):
            logger.info(f"  class {i}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        text = self.text_labels[label]
        
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"fail {img_path}: {e}")
            
            image = Image.new('RGB', (224, 224), color='white')
        
        return image, text, label

def collate_fn(batch):
    """define collate"""
    images, texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, texts, labels

class CLIPTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"use: {self.device}")
        
        ## wandb
        wandb.init(
            project="clip-osteoarthritis",
            config=vars(config),
            name=f"clip-training-{config.learning_rate}-{config.batch_size}"
        )
        
        
        self.load_model()
        
        
        self.create_datasets()
        
        
        self.create_optimizer()
        
       
        self.criterion = nn.CrossEntropyLoss()
        
        
        self._precompute_text_features()
        
        
        self.best_val_mae = float('inf')
        self.best_val_acc = 0.0
    
    
    def _precompute_text_features(self):
        """caclulate text features for all 5 classes"""
        logger.info("caclulate...")
        
       
        text_labels = [
            "No osteoarthritis, No radiographic features of osteoarthritis.",
            "Doubtful osteoarthritis, Doubtful narrowing of joint space and possible osteophytic lipping.",
            "Mild osteoarthritis, Definite osteophytes and possible narrowing of joint space.",
            "Moderate osteoarthritis, Multiple osteophytes, definite narrowing of joint space, some sclerosis, and possible deformity of bone ends.",
            "Severe osteoarthritis, Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
        ]
        
        
        with torch.no_grad():
            text_inputs = self.processor(
                text=text_labels,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        self.text_features = text_features  # shape: [5, hidden_size]
        logger.info(f"features: {self.text_features.shape}")
    
    def load_model(self):
        """CLIP"""
        logger.info(f"from {self.config.model_path} load...")
        try:
            self.model = CLIPModel.from_pretrained(self.config.model_path, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(self.config.model_path, local_files_only=True)
            self.model.to(self.device)
            logger.info("success!")
        except Exception as e:
            logger.error(f"fail: {e}")
            raise e
    
    def create_datasets(self):
        """create datasets"""
        
        
        train_dataset = OsteoarthritisDataset(
            data_root=self.config.train_data_path,
            processor=self.processor,
            is_train=True
        )
        
    
        val_dataset = OsteoarthritisDataset(
            data_root=self.config.val_data_path,
            processor=self.processor,
            is_train=False
        )
        
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"train: {len(train_dataset)}")
        logger.info(f"val: {len(val_dataset)}")
    
    def create_optimizer(self):
        """create"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
    
    def calculate_metrics(self, all_preds, all_labels):
        """caculae"""
       
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        
        total_correct = np.sum(all_preds == all_labels)
        total_acc = 100. * total_correct / len(all_labels)
        
        
        class_accs = []
        for i in range(5):
            mask = all_labels == i
            if np.sum(mask) > 0:
                class_acc = 100. * np.sum(all_preds[mask] == i) / np.sum(mask)
                class_accs.append(class_acc)
            else:
                class_accs.append(0.0)
        
        # MAE
        mae = np.mean(np.abs(all_preds - all_labels))
        
        return total_acc, class_accs, mae
    
    def plot_roc_curves(self, all_probs, all_labels, epoch, save_dir):
        """ROC"""
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        
        roc_dir = os.path.join(save_dir, 'roc_curves')
        os.makedirs(roc_dir, exist_ok=True)
        
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i in range(5):
            
            y_true = (all_labels == i).astype(int)
            y_score = all_probs[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for Each Class (Epoch {epoch+1})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(roc_dir, f'roc_5classes_epoch{epoch+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 0-1 vs 2-3-4
        plt.figure(figsize=(8, 8))
        
        
        y_true_binary = (all_labels >= 2).astype(int)
       
        y_score_binary = all_probs[:, 2:].sum(axis=1)
        
        fpr_binary, tpr_binary, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc_binary = auc(fpr_binary, tpr_binary)
        
        plt.plot(fpr_binary, tpr_binary, color='darkred', lw=2,
                label=f'Binary Classification (AUC = {roc_auc_binary:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Binary ROC Curve: (0-1) vs (2-3-4) (Epoch {epoch+1})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(roc_dir, f'roc_binary_epoch{epoch+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
       
        wandb.log({
            'roc/auc_class_0': auc(roc_curve((all_labels == 0).astype(int), all_probs[:, 0])[0],
                                  roc_curve((all_labels == 0).astype(int), all_probs[:, 0])[1]),
            'roc/auc_class_1': auc(roc_curve((all_labels == 1).astype(int), all_probs[:, 1])[0],
                                  roc_curve((all_labels == 1).astype(int), all_probs[:, 1])[1]),
            'roc/auc_class_2': auc(roc_curve((all_labels == 2).astype(int), all_probs[:, 2])[0],
                                  roc_curve((all_labels == 2).astype(int), all_probs[:, 2])[1]),
            'roc/auc_class_3': auc(roc_curve((all_labels == 3).astype(int), all_probs[:, 3])[0],
                                  roc_curve((all_labels == 3).astype(int), all_probs[:, 3])[1]),
            'roc/auc_class_4': auc(roc_curve((all_labels == 4).astype(int), all_probs[:, 4])[0],
                                  roc_curve((all_labels == 4).astype(int), all_probs[:, 4])[1]),
            'roc/auc_binary': roc_auc_binary,
            'epoch': epoch + 1
        })
    
    def plot_confusion_matrix(self, all_preds, all_labels, epoch):
        """Draw"""
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(5), yticklabels=range(5))
        plt.title(f'Confusion Matrix (Epoch {epoch+1})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_dir = os.path.join(self.config.save_dir, 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)
        plt.savefig(os.path.join(cm_dir, f'cm_epoch{epoch+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_epoch(self, epoch):
        """epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (images, texts, labels) in enumerate(progress_bar):
            
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
                size={"shortest_edge": 336}  
            )
            
            
            
            pixel_values = inputs['pixel_values'].to(self.device)
            labels = labels.to(self.device)
            
            
            self.optimizer.zero_grad()
            
            
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            
            logits = (image_features @ self.text_features.T) * self.model.logit_scale.exp()
            
            
            loss = self.criterion(logits, labels)
            
            
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            
            total_loss += loss.item()
            
            ##### logits
            predicted = logits.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            
            if batch_idx % 10 == 0:
                current_acc = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
                current_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'MAE': f'{current_mae:.3f}'
                })
        
      
        avg_loss = total_loss / len(self.train_loader)
        total_acc, class_accs, mae = self.calculate_metrics(all_preds, all_labels)
        
        
        logger.info(f'train Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={total_acc:.2f}%, MAE={mae:.3f}')
        logger.info(f'acc: {[f"{acc:.2f}%" for acc in class_accs]}')
        
        ## wandb
        wandb.log({
            'train/loss': avg_loss,
            'train/accuracy': total_acc,
            'train/mae': mae,
            'epoch': epoch + 1
        })
        
       
        for i, acc in enumerate(class_accs):
            wandb.log({f'train/class_{i}_accuracy': acc, 'epoch': epoch + 1})
        
        return avg_loss, total_acc, mae
    
    def validate(self, epoch):
        """val"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, texts, labels in tqdm(self.val_loader, desc="val"):
                
                inputs = self.processor(
                    images=images,  
                    return_tensors="pt",
                    padding=True
                )
                
                
            
                pixel_values = inputs['pixel_values'].to(self.device)
                labels = labels.to(self.device)
                
                
                image_features = self.model.get_image_features(pixel_values=pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
              
                logits = (image_features @ self.text_features.T) * self.model.logit_scale.exp()
                
              
                loss = self.criterion(logits, labels)
                
               
                total_loss += loss.item()
                
                
                predicted = logits.argmax(dim=1)
                probs = F.softmax(logits, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        
        avg_loss = total_loss / len(self.val_loader)
        total_acc, class_accs, mae = self.calculate_metrics(all_preds, all_labels)
        
        
        logger.info(f'val Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={total_acc:.2f}%, MAE={mae:.3f}')
        logger.info(f'acc: {[f"{acc:.2f}%" for acc in class_accs]}')
        
        ## wandb
        wandb.log({
            'val/loss': avg_loss,
            'val/accuracy': total_acc,
            'val/mae': mae,
            'epoch': epoch + 1
        })
        
       
        for i, acc in enumerate(class_accs):
            wandb.log({f'val/class_{i}_accuracy': acc, 'epoch': epoch + 1})
        
        # ROC
        self.plot_roc_curves(all_probs, all_labels, epoch, self.config.save_dir)
        
        
        self.plot_confusion_matrix(all_preds, all_labels, epoch)
        
        return avg_loss, total_acc, mae, all_probs, all_labels
    
    def save_model(self, epoch, val_mae, is_best=False):
        """save"""
        
        save_dir = os.path.join(self.config.save_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_mae': val_mae,
        }
        torch.save(checkpoint, os.path.join(save_dir, 'training_state.pth'))
        
        logger.info(f"save: {save_dir}")
        
        if is_best:
            best_dir = os.path.join(self.config.save_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_pretrained(best_dir)
            self.processor.save_pretrained(best_dir)
            
            
            best_info = {
                'epoch': epoch + 1,
                'val_mae': val_mae,
                'saved_at': os.path.join(best_dir)
            }
            with open(os.path.join(best_dir, 'best_model_info.json'), 'w') as f:
                json.dump(best_info, f, indent=2)
            
            logger.info(f"best: {best_dir} (MAE: {val_mae:.4f})")
    
    def train(self):
        """main"""
        logger.info("start...")
        
        for epoch in range(self.config.epochs):
            
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"Warmup epoch {epoch+1}, leraning rate: {warmup_lr:.2e}")
            
            
            train_loss, train_acc, train_mae = self.train_epoch(epoch)
            
            
            val_loss, val_acc, val_mae, val_probs, val_labels = self.validate(epoch)
            
            
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"learning rate: {current_lr:.2e}")
            
            
            wandb.log({
                'learning_rate': current_lr,
                'epoch': epoch + 1
            })
            
            # MAE
            is_best = val_mae < self.best_val_mae
            if is_best:
                self.best_val_mae = val_mae
                self.best_val_acc = val_acc
                logger.info(f"MAE: {val_mae:.4f}, Acc: {val_acc:.2f}%")
            
            
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_model(epoch, val_mae, is_best)
            
            logger.info("-" * 50)
        
        logger.info(f"val: {self.best_val_mae:.4f}, acc: {self.best_val_acc:.2f}%")
        
        
        wandb.finish()


class Config:
    def __init__(self):
        
        self.model_path = ""
        self.train_data_path = ""
        self.val_data_path = ""
        self.save_dir = ""
        self.batch_size = 32
        self.epochs = 120
        self.learning_rate = 1e-5
        self.weight_decay = 1e-5
        self.step_size = 30
        self.gamma = 0.5
        self.warmup_epochs = 5
        self.num_workers = 8
        self.save_interval = 1

def main():
    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
   
    config = Config()
    
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    
    trainer = CLIPTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()