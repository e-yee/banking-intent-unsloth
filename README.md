# Banking Intent Unsloth

## 1. Overview

This project aims to fine-tune `Qwen3-4B-Base` for BANKING77 intent classification.

Main pipeline:

1. Download the original BANKING77 dataset (using huggingface hub) and clean.
2. Fine-tune with Unsloth + LoRA
3. Run single-message inference and return the corresponding intent label

---

## 2. Project Structure

```text
banking-intent-unsloth/
|-- configs/
|   |-- train.yaml      # Settings for training        
|   \-- inference.yaml  # Settings for inference
|-- models/
|   |-- .gitignore  # Placeholder for fine-tuned models
|-- sample_data/
|   |-- .gitignore  # Placeholder for raw and preprocessed data
|-- scripts/
|   |-- data_collator.py    # Data collator for training
|   |-- preprocess_data.py  # Preprocess data script
|   |-- inference.py        # Load model and inference
|   \-- train.py            # Fine-tune model
|-- utils/
|   |-- logger.py # Customed logger
|   \-- paths.py  # Paths to folders
|-- example.env      # Store HuggingFace access token
|-- .gitignore       # Ignore pycache and .env    
|-- README.md        # Documents
|-- train.sh         # Shell script for training
|-- inference.sh     # Shell script for inferencing
\-- requirements.txt # Requirements   
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
git clone https://github.com/e-yee/banking-intent-unsloth.git
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

### 5.1 Download Dataset

```bash
python -m scripts.preprocess_data
```

### 5.2 Dataset Overview

The [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset contains **13,083** customer service queries across **77** banking intent categories.

| Split | File | Samples |
| ------- | ------ | --------- |
| Train | `sample_data/preprocessed/train.csv` | ~10,003 |
| Test | `sample_data/preprocessed/test.csv` | ~3,080 |

### 5.2 Expected CSV format

```csv
text,label
i am still waiting on my card?,12
what can i do if my card still hasn't arrived after 2 weeks?,12
...
```

### 5.3 Label files

- `sample_data/raw/categories.json` — ordered list of the 77 intent names
- `sample_data/preprocessedlabel_map.json` — `{ "numeric_id: "intent_name", ... }` mapping used during training and inference

### 5.4 How preprocessing works

`scripts/preprocess_data.py`:

1. Download raw dataset including `train.csv` and `test.csv`, extract categories into `categories.json`
2. Lower and trim extra spaces in text
3. Increment label number by one
4. Generate a mapping from numeric value to intent name, save into `label_map.json`

---

## 6. Training

### 6.1 Configuration (`configs/train.yaml`)

```yaml
model: # Settings for loading FastLanguageModel
  model_name: unsloth/Qwen3-4B-Base
  load_in_4bit: False
  max_seq_length: 2048
  dtype: null

lora: # Settings for adding LoRA adapter
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

sftconfig: # Settings for SFTTrainer arguments
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

tokenizer: # Settings for tokenizer's encoder
  return_tensors: pt
  padding: True
  truncation: True
  max_length: 2048

generate: # Settings for model's generation
  max_new_tokens: 2
  use_cache: True
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

This executes `python -m scripts.train`.

The fine-tuned is automatically saved to `models/`.

---

## 7. Evaluation Results

> 📊 *Run `bash train.sh` or execute `scripts/train.py` to populate these numbers.*

| Metric | Score |
| ---------- | ------- |
| Accuracy | 92.11% |
| Macro F1 | 92.11% |

**Test set:** `sample_data/preprocessed/test.csv` (~3,080 samples, 77 classes)

---

## 8. Inference

### 8.1 Configuration (`configs/inference.yaml`)

```yaml
model: # Settings for loading FastLanguageModel fine-tuned
  load_in_4bit: False
  max_seq_length: 2048
  dtype: null

tokenizer: # Settings for tokenizer's encoder
  return_tensors: pt
  padding: True
  truncation: True
  max_length: 2048

generate: # Settings for model's generation
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
message = "How do I locate my card?"
```

You can change this to any banking-related query by changing `MESSAGE`. The script will print the predicted intent label, e.g.:

```text
card_arrival
```

### 8.3 Using `IntentClassification` programmatically

```python
import yaml

from scripts.inference import IntentClassification
from utils.paths import BASE_DIR

with open(CONFIG_DIR / "inference.yaml", "r") as f:
    configs = yaml.safe_load(f)

configs["model"]["model_name"] = BASE_DIR / "models" / "unsloth" "Qwen3-4B-Base" / "finetuned"

message = "How do I locate my card?"
classifier = IntentClassification(configs)
classifier(message)
```

---

## 9. Demo

- [Kaggle Train and Evaluate](https://www.kaggle.com/code/ethanyee2706/banking-intent-unsloth-train)
- [Kaggle Inference](https://www.kaggle.com/code/ethanyee2706/banking-intent-unsloth-inference)
- [Video Demo](https://drive.google.com/drive/folders/1HEx5UYuCEEcvu6yQ5iocEiJKI6IwgDIp?usp=sharing)

---

## 10. References

- [BANKING77 Dataset — PolyAI](https://huggingface.co/datasets/PolyAI/banking77)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Unsloth Classification](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb)
