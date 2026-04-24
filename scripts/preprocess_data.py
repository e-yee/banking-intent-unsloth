import json
import polars as pl

from datasets import load_dataset, DatasetDict

from utils.paths import DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

def download_dataset(
    path: str, 
    revision: str
) -> tuple:
    """Download dataset with revision and convert to polars.DataFrames."""
    
    logger.info(f"Downloading dataset [{path} | revision={revision}]...")
    
    dataset: DatasetDict = load_dataset(
        path=path,
        revision=revision
    )
    
    output_path = DATA_DIR / "raw"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Converts to Polars
    train_pl = pl.from_arrow(dataset["train"].data.table)
    logger.info(f"Loaded {train_pl.shape[0]:,} rows (train data).")
    
    test_pl = pl.from_arrow(dataset["test"].data.table)
    logger.info(f"Loaded {test_pl.shape[0]:,} rows (test data).")
    
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
    
    train_pl.write_csv(train_path)
    test_pl.write_csv(test_path)
    
    logger.success("Dataset saved successfully.")
    
    return train_pl, test_pl, labels

def clean_text(text: str) -> str:
    """Lower case and strip extra spaces of the given text."""
    return " ".join(text.lower().split())
    
def preprocess_data(
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    labels: list
) -> tuple:
    """Preprocess raw data."""
    
    label_map = {f"L<{i}>":label for i, label in enumerate(labels)}
    mapping = {i:f"L<{i}>" for i in range(len(labels))}
    
    def preprocess(data: pl.DataFrame) -> pl.DataFrame:
        return (
            data
            .with_columns(
                pl.col("text")
                .map_elements(clean_text, return_dtype=pl.String)
                .alias("preprocessed_text"),
                
                pl.col("label")
                .replace_strict(mapping, return_dtype=pl.String)
                .alias("label_encoded")
            )
            .select(["preprocessed_text", "label_encoded"])
        )
    
    # Preprocess data
    logger.info("Preprocessing dataset...")
    
    preprocessed_train_data = preprocess(train_data)
    preprocessed_test_data = preprocess(test_data)
    
    # Save data
    output_path = DATA_DIR / "preprocessed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving preprocessed dataset to {output_path}...")
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    preprocessed_train_data.write_csv(train_path)
    preprocessed_test_data.write_csv(test_path)
    
    with open(output_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    logger.success("Preprocessed dataset saved successfully.")
    
    return preprocessed_train_data, preprocessed_test_data, label_map

def main():
    try:
        train_pl, test_pl, labels = download_dataset("PolyAI/banking77", "refs/convert/parquet")
        preprocess_data(train_pl, test_pl, labels)
    except Exception as e:
        logger.error(f"Preprocessing data failed: {e}")
    
if __name__ == "__main__":
    main()