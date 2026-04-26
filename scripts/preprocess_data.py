import os
import json
import polars as pl

from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from typing import Dict, List, Tuple

from utils.paths import DATA_DIR
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

def download_dataset(
    path: str, 
    revision: str
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str] | None]:
    """Download dataset with revision and convert to polars.DataFrames."""
    
    logger.info(f"Downloading dataset [{path} | revision={revision}]...")
    
    hf_token = os.getenv("HF_TOKEN")
    dataset: DatasetDict = load_dataset(
        path=path,
        revision=revision,
        token=hf_token
    )
    
    output_path = DATA_DIR / "raw"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Converts to Polars
    train_df = pl.from_arrow(dataset["train"].data.table)
    logger.info(f"Loaded {train_df.shape[0]:,} rows (train data).")
    
    test_df = pl.from_arrow(dataset["test"].data.table)
    logger.info(f"Loaded {test_df.shape[0]:,} rows (test data).")
    
    # Labels
    label_feature = dataset["train"].features.get("label")
    
    labels = None
    if label_feature:
        labels = label_feature.names
        logger.info(f"Loaded {len(labels)} labels.")
        
        categories_path = output_path / "categories.json"
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved labels to {categories_path}.")
    else:
        logger.warning("No labels found.")
    
    # Save dataset
    logger.info(f"Saving dataset to {output_path}...")
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_df.write_csv(train_path)
    test_df.write_csv(test_path)
    
    logger.success("Dataset saved successfully.")
    
    return train_df, test_df, labels

def clean_text(text: str) -> str:
    """Lower case and strip extra spaces of the given text."""
    return " ".join(text.lower().split())
    
def preprocess_data(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    labels: list
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, str]]:
    """Preprocess raw data."""
    
    label_map = {i+1:label for i, label in enumerate(labels)}
    
    def preprocess(data: pl.DataFrame) -> pl.DataFrame:
        return (
            data
            .with_columns(
                pl.col("text")
                .map_elements(clean_text, return_dtype=pl.String),
                
                pl.col("label") + 1
            )
        )
    
    # Preprocess data
    logger.info("Preprocessing dataset...")
    
    preprocessed_train_df = preprocess(train_df)
    preprocessed_test_df = preprocess(test_df)
    
    # Save data
    output_path = DATA_DIR / "preprocessed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving preprocessed dataset to {output_path}...")
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    preprocessed_train_df.write_csv(train_path)
    preprocessed_test_df.write_csv(test_path)
    
    with open(output_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    logger.success("Preprocessed dataset saved successfully.")
    
    return preprocessed_train_df, preprocessed_test_df, label_map

def main():
    try:
        train_pl, test_pl, labels = download_dataset("PolyAI/banking77", "refs/convert/parquet")
        preprocess_data(train_pl, test_pl, labels)
    except Exception as e:
        logger.error(f"Preprocessing data failed: {e}")
    
if __name__ == "__main__":
    main()