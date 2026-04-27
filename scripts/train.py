import yaml
import torch
import datasets
import polars as pl

from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Tuple
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, classification_report

from utils.logger import get_logger
from utils.paths import BASE_DIR, CONFIG_DIR, DATA_DIR
from data_collator import DataCollatorForTwoLastTokensLM

import transformers
transformers.logging.set_verbosity_error()

logger = get_logger(__name__)

PROMPT_TEMPLATE  = """Here is a banking intent:
{}

Classify this banking intent into one label:
01 to 77

SOLUTION
{}"""

def load_config(path: Path) -> Dict[str, Dict[str, str]]:
    """Load configs."""
    
    logger.info(f"Loading configs from {path}...")
    
    with open(path / "train.yaml", "r") as f:
        data = yaml.safe_load(f)
        logger.info(f"Loaded {len(data)} configs.")
    
    logger.success("Configs loaded successfully.")
    
    return data

def load_data(path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load preprocessed dataset."""
    
    logger.info(f"Loading data from {path}...")
    
    train_path = path / "train.csv"
    test_path = path / "test.csv"
    label_path = path / "label_map.json"
    
    train_df = pl.read_csv(train_path)
    logger.info(f"Loaded {train_df.shape[0]:,} rows (train data).")
    
    test_df = pl.read_csv(test_path)
    logger.info(f"Loaded {test_df.shape[0]:,} rows (test data).")
        
    logger.success(f"Data loaded successfully.")

    return train_df, test_df

def format_train_data(
    prompt: str,
    df: pl.DataFrame
) -> datasets.Dataset:
    """Format data like prompt template."""
    
    logger.info("Formatting data...")
    
    prompt_mapping = {}
    for row in df.iter_rows(named=True):
        text_ = row["text"]
        label_ = f"{row["label"]:02d}"
        prompt_mapping[text_] = prompt.format(text_, label_)
    
    prompt_df = df.with_columns(
        pl.col("text")
        .replace_strict(prompt_mapping, return_dtype=pl.String)
    )
    
    dataset = datasets.Dataset.from_polars(prompt_df)
    logger.info(f"Formatted {dataset.shape[0]:,} rows (train data).")
    
    logger.success("Data formatted successfully.")
    
    return dataset

def build_model_and_tokenizer(
    config: Dict[str, Any],
    max_digit: int = 10 # For finetuning LM head
) -> Tuple[Any, Any]:
    """Load and build model for finetuning two last tokens LM."""

    logger.info(f"Loading model and tokenizer [{config['model']['model_name']} | load_in_4bit={config['model']['load_in_4bit']}]..")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        device_map={"": 0},
        **config
    )
    
    logger.success(f"Loaded model and tokenizer successfully.")
    
    # logger.info("Modifying the LM head for finetuning...")
    
    # lm_head = model.lm_head
    # device = lm_head.weight.device

    # number_token_ids = [tokenizer.convert_tokens_to_ids(str(i))
    #                     for i in range(max_digit)]
    # used_token_ids = torch.tensor(number_token_ids, device=device)
    
    # logger.info(f"Used token indices: {number_token_ids}.")
    
    # logger.info(f"Setting requires grad for parameters of used token indices...")
    
    # grad_mask = torch.zeros_like(lm_head.weight)
    # grad_mask[used_token_ids, :] = 1.0
    # lm_head.weight.requires_grad = True
    # lm_head.weight.register_hook(lambda g: g * grad_mask)
    
    # if lm_head.bias is not None:
    #     bias_mask = torch.zeros_like(lm_head.bias)
    #     bias_mask[used_token_ids] = 1.0
    #     lm_head.bias.requires_grad = True
    #     lm_head.bias.register_hook(lambda g: g * bias_mask)
        
    # trainable_params = sum(p.numel() for p in lm_head.parameters() if p.required_grads)
    # logger.info(f"Trainable parameters: {trainable_params}.")
    
    # logger.success("Builded model successfully.")
    
    return model, tokenizer

def add_lora_adapter(model, config: Dict[str, Any]) -> Any:
    """Add lora adapter for model."""
    
    logger.info("Adding lora adapter for model...")
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        **config
    )
    
    logger.success("Added lora adapter successfully.")
    
    return model

def build_trainer(
    model, 
    tokenizer,
    config: Dict[str, Any],
    data_collator: DataCollatorForTwoLastTokensLM,
    train_dataset: datasets.Dataset
) -> Any:
    """Build trainer."""
    
    logger.info("Building trainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=BASE_DIR / "models" / model.config._name_or_path / "checkpoints",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            dataset_text_field="text",
            **config
        ),
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    logger.success("Builded trainer successfully.")
    
    return trainer

def format_test_data(
    test_df: pl.DataFrame,
    tokenizer: Any
) -> pl.DataFrame:
    """Format test data for evaluation."""
    
    logger.info("Formatting test data...")
    
    
    test_df = test_df.with_columns(
        pl.col("text")
        .map_elements(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
        .alias("token_length")
    ).sort("token_length")
    
    logger.success("Formatted test data successfully.")
    
    return test_df

def evaluate(
    model: Any,
    tokenizer: Any,
    test_df: pl.DataFrame,
    configs: Dict[str, Any]
):
    """Evaluate model on test data."""
    
    logger.info("Evaluating model...")
    
    FastLanguageModel.for_inference(model)
    
    batch_size = 16
    device = model.device
    
    test_df = format_test_data(test_df, tokenizer)
    evaluate_prompt = PROMPT_TEMPLATE.split("SOLUTION\n{}")[0] + "SOLUTION\n"
    
    pred_labels = []
    true_labels = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(test_df), batch_size), desc="Evaluating"):
            batch = test_df[i:i+batch_size]
            prompts = [evaluate_prompt.format(text) for text in batch["text"]]
            inputs = tokenizer(
                prompts,
                **configs["tokenizer"],
            ).to(device)
            outputs = model.generate(**inputs, **configs["generate"])
            
            preds = list(map(int, tokenizer.batch_decode(outputs[:, -2:])))
            truth = batch["label"].to_list()
            
            pred_labels.extend(preds)
            true_labels.extend(truth)
        
    logger.info(f"Accuracy: {accuracy_score(true_labels, pred_labels):.3f}")
    logger.info(f"\nReport\n: {classification_report(true_labels, pred_labels)}" )
    
    logger.success("Evaluated successfully.")
    
def save_model_and_tokenizer(
    model,
    tokenizer,
    path: Path
):
    """Save model and tokenizer to path."""
    
    logger.info(f"Saving model and tokenizer to {path}...")
    
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    logger.success("Saved model and tokenizer successfully.")

def main():
    try:
        configs = load_config(CONFIG_DIR)
        train_df, test_df, label_map = load_data(DATA_DIR / "preprocessed")
        train_dataset = format_train_data(PROMPT_TEMPLATE, train_df)
        
        model, tokenizer = build_model_and_tokenizer(configs["model"])
        model = add_lora_adapter(model, configs["lora"])
        collator = DataCollatorForTwoLastTokensLM(tokenizer=tokenizer)
        
        trainer = build_trainer(model, tokenizer, configs["sftconfig"], collator, train_dataset)
        trainer_stats = trainer.train()
        print(trainer_stats)
        
        evaluate(model, tokenizer, test_df)
        
        path = BASE_DIR / "models" / model.config._name_or_path / "finetuned"
        save_model_and_tokenizer(model, tokenizer, path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
    
if __name__ == "__main__":
    main()
