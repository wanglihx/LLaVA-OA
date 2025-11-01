from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
import os
import glob
from tqdm import tqdm
import csv
import json
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """load"""
    try:
        print(f"load: {model_path}")
        model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        print("successfully loaded local model")
        return model, processor
    except Exception as e:
        print(f"fail: {e}")
        print("check path please")
        raise e

def precompute_text_features(model, processor, device):
    """calculate text features"""
    print("caculate...")
    
    
    text_labels = [
        "No osteoarthritis, No radiographic features of osteoarthritis.",
        "Doubtful osteoarthritis, Doubtful narrowing of joint space and possible osteophytic lipping.",
        "Mild osteoarthritis, Definite osteophytes and possible narrowing of joint space.",
        "Moderate osteoarthritis, Multiple osteophytes, definite narrowing of joint space, some sclerosis, and possible deformity of bone ends.",
        "Severe osteoarthritis, Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends."
    ]
    
    
    with torch.no_grad():
        text_inputs = processor(
            text=text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    print(f"text: {text_features.shape}")
    
    return text_features.to(device), text_labels

def load_image(image_path):
    """load"""
    try:
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"fail {image_path}: {e}")
        return None

def get_test_data(test_directory):
    """label"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    all_images = []
    all_labels = []
    
    
    for class_id in range(5):
        class_dir = os.path.join(test_directory, str(class_id))
        if not os.path.exists(class_dir):
            print(f"not found {class_dir}")
            continue
            
        class_images = []
        for extension in image_extensions:
            class_images.extend(glob.glob(os.path.join(class_dir, extension)))
            class_images.extend(glob.glob(os.path.join(class_dir, extension.upper())))
        
        print(f"class {class_id}: get {len(class_images)} images")
        
        for img_path in class_images:
            all_images.append(img_path)
            all_labels.append(class_id)
    
    return all_images, all_labels

def calculate_metrics(all_preds, all_labels):
    """caculate"""
    
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

def plot_roc_curves(all_probs, all_labels, save_dir):
    """ROC"""
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    
    roc_dir = os.path.join(save_dir, 'roc_curves')
    os.makedirs(roc_dir, exist_ok=True)
    
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    auc_scores = []
    for i in range(5):
        
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'Class {i} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(roc_dir, 'roc_5classes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 0-1 vs 2-3-4
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
    plt.title('Binary ROC Curve: (0-1) vs (2-3-4)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(roc_dir, 'roc_binary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAUC:")
    for i, auc_score in enumerate(auc_scores):
        print(f"  class {i}: {auc_score:.3f}")
    print(f"  0-1 vs 2-3-4: {roc_auc_binary:.3f}")

def plot_confusion_matrix(all_preds, all_labels, save_dir):
    """draw"""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=range(5), yticklabels=range(5))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_dir = os.path.join(save_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    plt.savefig(os.path.join(cm_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 
    print("\nresult:")
    print(cm)

def save_results(results, output_file):
    """save"""
    
    json_file = output_file + '_detailed.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nresult: {json_file}")
    
    
    csv_file = output_file + '_summary.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true', 'predict', 'confidence', 'distribution'])
        
        for result in results:
            probs_str = '; '.join([f"class{i}:{v:.4f}" for i, v in enumerate(result['probabilities'])])
            writer.writerow([
                result['image_path'],
                result['true_label'],
                result['predicted_label'],
                f"{result['confidence']:.4f}",
                probs_str
            ])
    print(f"CSV: {csv_file}")

def evaluate_model(model, processor, text_features, test_directory, output_dir, device):
    """eval"""
    model = model.to(device)
    model.eval()
    
    
    text_features = text_features.to(device)
    
    
    image_paths, true_labels = get_test_data(test_directory)
    
    if not image_paths:
        print("not found")
        return
    
    print(f"\nget {len(image_paths)} images")
    
    
    all_preds = []
    all_labels = []
    all_probs = []
    results = []
    
    
    for idx, (image_path, true_label) in enumerate(tqdm(zip(image_paths, true_labels), 
                                                       total=len(image_paths), 
                                                       desc="progress")):
        # load
        image = load_image(image_path)
        if image is None:
            continue
        
        
        inputs = processor(
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        pixel_values = inputs['pixel_values'].to(device)
        
        
        with torch.no_grad():
            
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            
            
            logit_scale = model.logit_scale.exp()
            logits = (image_features @ text_features.T) * logit_scale
            
            
            probs = F.softmax(logits, dim=1)
            predicted = logits.argmax(dim=1)
        
        probs_list = probs.cpu().numpy()[0]
        pred_label = predicted.item()
        
        
        all_preds.append(pred_label)
        all_labels.append(true_label)
        all_probs.append(probs_list)
        
        
        result = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'true_label': int(true_label),  
            'predicted_label': int(pred_label),  
            'confidence': float(probs_list[pred_label]),  
            'probabilities': [float(p) for p in probs_list]  
        }
        results.append(result)
    
    
    total_acc, class_accs, mae = calculate_metrics(all_preds, all_labels)
    
    
    print("\n" + "="*50)
    print("result")
    print("="*50)
    print(f"acc: {total_acc:.2f}%")
    print(f"MAE: {mae:.3f}")
    print("\nacc per:")
    for i, acc in enumerate(class_accs):
        print(f"  class {i}: {acc:.2f}%")
    
    
    print("\nreport:")
    print(classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(5)]))
    
    
    plot_confusion_matrix(all_preds, all_labels, output_dir)
    
    
    plot_roc_curves(all_probs, all_labels, output_dir)
    
    
    output_file = os.path.join(output_dir, 'evaluation_results')
    save_results(results, output_file)
    
    
    summary = {
        'total_images': len(all_preds),
        'total_accuracy': float(total_acc),  
        'mae': float(mae),  
        'class_accuracies': {f'class_{i}': float(acc) for i, acc in enumerate(class_accs)},  
        'model_path': model_path,
        'test_directory': test_directory
    }
    
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nsave: {output_dir}")

# main
if __name__ == "__main__":
    
    model_path = ""
    test_directory = ""
    output_directory = ""
    
    
    os.makedirs(output_directory, exist_ok=True)
    
    print("load...")
    model, processor = load_model(model_path)
    print("done")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use: {device}")
    
    
    model = model.to(device)
    
    
    text_features, text_labels = precompute_text_features(model, processor, device)
    
    print("\nstart...")
    evaluate_model(model, processor, text_features, test_directory, output_directory, device)
    
    print("\ndone!")