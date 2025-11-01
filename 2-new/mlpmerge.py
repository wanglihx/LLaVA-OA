#!/usr/bin/env python3
"""
MLP merge
"""

import torch
import json
import os
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import argparse

def integrate_mlp_weights():
    
    base_model_path = ""
    mlp_checkpoint_path = ""
    output_path = ""
    
    print("="*70)
    print("Merge...")
    print("="*70)
    print(f"📁 : {base_model_path}")
    print(f"📁 : {mlp_checkpoint_path}")
    print(f"📁 : {output_path}")
    print("="*70)
    
    
    if not Path(base_model_path).exists():
        print(f"❌ : {base_model_path}")
        return False
    
    mlp_file = Path(mlp_checkpoint_path) / "mm_projector.bin"
    if not mlp_file.exists():
        print(f"❌ : {mlp_file}")
        return False
    
    
    os.makedirs(output_path, exist_ok=True)
    print(f"\n✅ : {output_path}")
    
    
    print("\n📋 loade...")
    mlp_weights = torch.load(mlp_file, map_location='cpu')
    
    print("  📊 MLP:")
    for key, value in mlp_weights.items():
        print(f"    - {key}: {list(value.shape)} (dtype: {value.dtype})")
    
    
    print("\n📋 copy...")
    
    
    essential_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "tokenizer.json",
        "README.md"
    ]
    
    
    for file in essential_files:
        src = Path(base_model_path) / file
        dst = Path(output_path) / file
        if src.exists():
            print(f"  📄 : {file}")
            shutil.copy2(src, dst)
    
    
    for i in [1, 2, 4]:  
        filename = f"model-0000{i}-of-00004.safetensors"
        src = Path(base_model_path) / filename
        dst = Path(output_path) / filename
        if src.exists():
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"  📄 copy: {filename} ({size_mb:.1f} MB)")
            shutil.copy2(src, dst)
    
    
    print("\n📋  model-00003-of-00004.safetensors...")
    
   
    model_003_path = Path(base_model_path) / "model-00003-of-00004.safetensors"
    print(f"  📖 : {model_003_path.name}")
    
    
    tensors = {}
    with safe_open(model_003_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            
    print(f"  📊  {len(tensors)} ")
    
    
    original_mlp_keys = [k for k in tensors.keys() if 'mm_projector' in k]
    if original_mlp_keys:
        print(f"  📊 :")
        for key in original_mlp_keys:
            print(f"    - {key}: {list(tensors[key].shape)}")
    
    
    print("\n  🔄 ...")
    replaced_count = 0
    for key, value in mlp_weights.items():
        if key in tensors:
            old_shape = tensors[key].shape
            new_shape = value.shape
            
            
            if old_shape != new_shape:
                print(f"    ⚠️  : {key} ")
                print(f"       : {old_shape}, new: {new_shape}")
                continue
            
            
            if tensors[key].dtype != value.dtype:
                value = value.to(tensors[key].dtype)
                print(f"    📝  {key} to {tensors[key].dtype}")
            
            tensors[key] = value
            print(f"    ✅ : {key}")
            replaced_count += 1
        else:
            print(f"    ⚠️  {key}")
    
    print(f"\n  ✅  {replaced_count} ")
    
    
    output_model_003 = Path(output_path) / "model-00003-of-00004.safetensors"
    print(f"\n  💾 : {output_model_003.name}")
    
   
    metadata = {"format": "pt"}
    save_file(tensors, output_model_003, metadata=metadata)
    
    size_mb = output_model_003.stat().st_size / 1024 / 1024
    print(f"  ✅  ({size_mb:.1f} MB)")
    
    
    print("\n📋 val...")
    
    index_file = Path(output_path) / "model.safetensors.index.json"
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    
    weight_map = index_data.get("weight_map", {})
    mlp_keys_in_index = [k for k in weight_map.keys() if 'mm_projector' in k]
    
    if mlp_keys_in_index:
        print("  📊 :")
        for key in mlp_keys_in_index:
            print(f"    - {key} -> {weight_map[key]}")
        
        
        all_correct = all(weight_map[k] == "model-00003-of-00004.safetensors" 
                         for k in mlp_keys_in_index)
        if all_correct:
            print("  ✅ ")
        else:
            print("  ⚠️ ")
    
    
    print("\n📋 ...")
    
    verify_script = f"""#!/usr/bin/env python3
'''check'''
import torch
from safetensors import safe_open

# check
model_file = "{output_path}/model-00003-of-00004.safetensors"
with safe_open(model_file, framework="pt", device="cpu") as f:
    print("merge:")
    for key in f.keys():
        if 'mm_projector' in key:
            tensor = f.get_tensor(key)
            print(f"  ✓ {{key}}: {{tensor.shape}}")

print("\\n✅ ")
"""
    
    verify_script_path = Path(output_path) / "verify_model.py"
    with open(verify_script_path, 'w') as f:
        f.write(verify_script)
    os.chmod(verify_script_path, 0o755)
    print(f"  ✅ verify_model.py")
    
   
    print("\n📋 ...")
    
    info_content = f"""# Merge

model = AutoModelForCausalLM.from_pretrained(
    "{output_path}",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("{output_path}")

"""
    
    with open(Path(output_path) / "MODEL_INFO.md", 'w') as f:
        f.write(info_content)
    print("  ✅ ")
    
   
    print("\n" + "="*70)
    print("🎉 ！")
    print(f"📁 : {output_path}")
    print("="*70)
    
    
    print("\n📂:")
    output_files = sorted(Path(output_path).glob("*"))
    total_size = 0
    for file in output_files:
        size = file.stat().st_size / 1024 / 1024
        total_size += size
        if file.suffix in ['.safetensors', '.bin']:
            print(f"  - {file.name:<40} {size:>8.1f} MB")
        else:
            print(f"  - {file.name}")
    
    print(f"\n📊: {total_size/1024:.2f} GB")
    
    
    print("\n🔍")
    os.system(f"cd {output_path} && python3 verify_model.py")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Merge")
    parser.add_argument("--base-model", type=str,
                       default="",
                       help="base model path")
    parser.add_argument("--mlp-path", type=str,
                       default="",
                       help="mlp path")
    parser.add_argument("--output", type=str,
                       default="",
                       help="output path")
    
    args = parser.parse_args()
    
    success = integrate_mlp_weights()
    
    if success:
        print("\n✅ ")
        
    else:
        print("\n❌ ")
        exit(1)

if __name__ == "__main__":
    main()