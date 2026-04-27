# Banking Intent Unsloth

## Table of Contents

- [1. Overview](#1-overview)
- [2. Project Structure](#2-project-structure)
- [Training](#training)
- [Evaluation](#evaluation)


## 1. Overview

This project aims to fine-tune `Qwen3-4B-Base` for BANKING77 intent classification.

Main pipeline:

1. Download the original BANKING77 dataset (using the `.csv` in git)
2. Clean and split the dataset into train/validation/test
3. Fine-tune with Unsloth + LoRA
4. Run single-message inference and return the corresponding intent label

---

## 2. Project Structure

```text
banking-intent-unsloth/
|-- configs/
|   |-- train.yaml          
|   \-- inference.yaml
|-- models/
|   |-- .gitignore
|-- sample_data/
|   |-- .gitignore
|-- scripts/
|   |-- data_collator.py 
|   |-- preprocess_data.py
|   |-- inference.py      
|   \-- train.py          
|-- train.sh              
|-- inference.sh          
|-- requirements.txt      
\-- README.md
```

---

## 3. System Requirements

| Component | Requirement |
| ----------- | ------------- |
| **OS** | Linux / Windows (WSL recommended for training) |
| **Python** | 3.13 |
| **GPU** | NVIDIA GPU with ≥ 14 GB VRAM (e.g. RTX 3090 / A100) |
| **CUDA** | 12.1+ |
| **Disk** | ~5 GB free (model weights + dataset) |

> **Note:** Unsloth requires a CUDA-capable GPU. CPU-only environments are **not** supported for training or inference.

---

## 4. Environment Setup

### 4.1 Clone the repository

```bash
git clone https://github.com/<your-username>/banking-intent-unsloth.git
cd banking-intent-unsloth
```

### 4.2 Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Data Preparation

Run training to automatically download and preprocess dataset.

### 5.1 Dataset overview

The [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset contains **13,083** customer service queries across **77** banking intent categories.

| Split | File | Samples |
| ------- | ------ | --------- |
| Train | `sample_data/preprocessed/train.csv` | ~10,003 |
| Test | `sample_data/preprocessed/test.csv` | ~3,080 |

### 5.2 Expected CSV format

```csv
text,label
"What is the exchange rate?",exchange_rate
"I lost my card.",lost_or_stolen_card
...
```

### 5.3 Label files

- `sample_data/raw/categories.json` — ordered list of the 77 intent names
- `sample_data/preprocessedlabel_map.json` — `{ "numeric_id: "intent_name", ... }` mapping used during training and inference

### 5.4 How preprocessing works

`scripts/preprocess_data.py`:

1. Loads a CSV with `datasets.load_dataset`
2. Maps each `category` string to a numeric label via `map.json`
3. Performs a stratified 90/10 train/validation split (seed `42`)

---

## 6. Training

### 6.1 Configuration (`configs/train.yaml`)

```yaml
model_name: unsloth/Qwen3-4B-Base
load_in_4bit: False
max_seq_length: 2048
dtype: null
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
warmup_steps: 15
learning_rate: 2e-4
packing: False
logging_steps: 10
optim: adamw_8bit
weight_decay: 0.01
lr_scheduler_type: cosine
seed: 3407
num_train_epochs: 1
report_to: none

# LoRA
r: 16
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj 
lora_alpha: 16
lora_dropout: 0
bias: none
use_gradient_checkpointing: unsloth
random_state: 3407
use_rlora: True
```

Adjust these values before running if needed.

### 6.2 Prompt template

```text
Here is a banking intent:
{}

Classify this banking intent into one label:
01 to 77

SOLUTION
{}
```

Only the tokens after `SOLUTION` contribute to the loss (prefix tokens are masked with `-100`).

### 6.3 Run training

```bash
bash train.sh
```

This executes `python -m scripts.preprocess_data && python -m scripts.train`.

The fine-tuned is automatically saved to `models/`.

---

## 7. Evaluation Results

> 📊 *Run `bash train.sh` (which also calls `_evaluate`) or execute `scripts/train.py` to populate these numbers.*

| Metric | Score |
| ---------- | ------- |
| Accuracy | 92.11% |
| Macro F1 | 92.11% |

**Test set:** `sample_data/test.csv` (~3,080 samples, 77 classes)

---

## 8. Inference

### 8.1 Configuration (`configs/inference.yaml`)

```yaml
model:
  load_in_4bit: False
  max_seq_length: 2048
  dtype: null

tokenizer:
  return_tensors: pt
  padding: True
  truncation: True
  max_length: 2048

generate:
  max_new_tokens: 2
  use_cache: True
```

### 8.2 Run inference

```bash
bash inference.sh --message MESSAGE
```

This executes `python -m scripts.inference "$@"`.

The default message in `inference.py` is:

```python
message = "Am I able to get a card in EU?"
```

You can change this to any banking-related query. The script will print the predicted intent label, e.g.:

```text
card_abroad
```

### 8.3 Using `IntentClassification` programmatically

```python
from scripts.inference import IntentClassification
from pathlib import Path

classifier = IntentClassification(
    model_path=Path("path/to/model"),
    yaml_path=Path("path/to/config")
)

intent = classifier("Am I able to get a card in EU?")
print(intent)  # → "country_support"
```

---

## 9. Demo

> 🎬 **Demo video coming soon.**
> **Please note that my laptop doesn't have a GPU so my demo will be on Kaggle**
> **Link to Kaggle inference and Evaluation notebook: **
> *(Place your demo video or GIF here — e.g., `![Demo](assets/demo.gif)` or a YouTube link.)*

---

## 10. References

- [BANKING77 Dataset — PolyAI](https://huggingface.co/datasets/PolyAI/banking77)
- [Unsloth](https://github.com/unslothai/unsloth)
