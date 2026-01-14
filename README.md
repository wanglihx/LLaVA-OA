# README

## Preprint
https://arxiv.org/abs/2601.02443

## Base Environment

- **PyTorch**: 2.1.2
- **Python**: 3.10 (Ubuntu 22.04)
- **CUDA**: 11.8

> **Critical:** **verify every model path, dataset path, and output/input path before running any command.**

We have repeated the following operations on a new server to ensure full reproducibility of the study.

## Directory Map & Path Notes

- `env/`: replace every placeholder path inside the environment YAML and TXT files with the actual locations on your server.
- `1-copy/`: provides patched files that must overwrite the same-named files inside the cloned LLaVA repository.
- `2-new/`: contains new scripts that need to be copied into the LLaVA repository.
- `3-clip/`: holds CLIP training, evaluation, and visualization scripts.
- `4-evalresult/`: stores original evaluation outputs that have already been renamed.
- `5-dataset/`: contains dataset JSON templates and the minimal image dataset file structure; choose the files that match the dataset size you intend to use.
- `6-apieval/`: includes evaluation files for closed-source MLLMs

Follow the steps below to reproduce the experiments while keeping every path consistent with your infrastructure.

## 1. Preparation

### 1.1 Clone and Patch LLaVA

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

Replace the following files in the cloned repository with the versions from `1-copy/`; you can also directly copy and paste the code content to see the changes made.

- `llava/serve/gradio_web_server.py`
- `llava/train/train_mem.py`
- `llava/train/train.py`
- `scripts/zero3.json`
- `scripts/v1_5/finetune_task_lora.sh`
- `llava/serve/examples/`

Add the new scripts from `2-new/` to the specified paths:

- `prompteval.py` -> `llava/serve/`
- `mlpmerge.py`, `mlp-train.sh` -> `scripts/`

### 1.2 Environment Setup

1. **LLaVA environment**
   ```bash
   cd LLaVA
   conda env create -f /yourpath/env/environment-llava-conda.yml
   pip install -r /yourpath/env/environment-llava-pip.txt
   ```
2. **CLIP environment**
   ```bash
   cd LLaVA
   conda env create -f /yourpath/env/environment-clip-conda.yml
   pip install -r /yourpath/env/environment-clip-pip.txt
   ```

### 1.3 Datasets and Models

- JSON instruction-image pairs: organise them under `Dataset/` and ensure the scripts point to the correct files.
- Image dataset download: [https://data.mendeley.com/datasets/56rmx5bjcr/1](https://data.mendeley.com/datasets/56rmx5bjcr/1)
- Minimal image dataset file structure: `Dataset/image_structure`
- Base multimodal model (LLaVA-Med v1.5): [https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)

## 2. Training

### 2.1 LLaVA Training Workflows

```bash
cd LLaVA
conda activate llava
```

- Adjust the training scripts according to the strategy described in the paper before launching them.
- LoRA finetuning:
  ```bash
  bash finetune_task_lora.sh
  ```
- MLP training:
  ```bash
  bash mlp-train.sh
  ```
- Select the dataset JSON files you need from `5-Dataset/` and confirm every path inside the scripts points to those files.
- After standalone MLP training, merge the weights to build the final model:
  ```bash
  python mlpmerge.py
  ```

## 3. Evaluation

### 3.1 Closed-Source Model Evaluation

- The evaluation files are located in the `6-apieval` directory. Use the script `apieval.py` and fill in the information for the models you wish to evaluate. The image URLs are read from `url.csv`, for which we have provided an example. Note that GPT-5 only supports setting the maximum token parameter; we recommend setting `max_token` to 2048 for GPT-5 and Gemini-2.5-Pro due to their built-in reasoning modes, and 512 for all other models.

### 3.2 Other MLLMs Evaluation

```bash
cd LLaVA
conda activate llava
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

**LoRA-adapted model worker**

```bash
python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path /yourpath \
    --model-base /yourpath
```

**Full-parameter model worker**

```bash
python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path /yourpath
```

**Sanity check**

```bash
python -m llava.serve.test_message \
    --model-name llava-med-v1.5-mistral-7b \
    --controller http://localhost:10000
```

**Batch evaluation**

```bash
PYTHONPATH=/yourpath/autodl-tmp/LLaVA:$PYTHONPATH \
python /yourpath/LLaVA/llava/serve/prompteval.py \
    --controller-url http://localhost:10000 \
    --batch \
    --grades all \
    --batch-temperature 0.01
```

## 4. CLIP Training, Evaluation, and Visualisation

- Base vision tower checkpoint: [https://huggingface.co/openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
- Activate the environment:
  ```bash
  conda activate clip
  ```
- Scripts inside `3-clip/`:
  - Training: `python cliptrain.py`
  - Evaluation: `python clipeval.py`
  - Visualisation (Grad-CAM): `python gradcam.py`

Ensure each script references the correct dataset, checkpoint, and output directories before you launch it.

## 5. Original Data

- All evaluation summaries are stored under `4-evalresult/`.
- `clipeval.py` produces `evaluation_results_detailed.json`.
- `prompteval.py` and the closed-source evaluation script output files named `2025-xx-xx-summaryxxx.json`.
- The files produced above are renamed according to the model names and stored in the `4-evalresult/` directory.

## 6. Model Weights

- The CLIP-OA weights are available at https://huggingface.co/wanglihx/CLIP-OA







