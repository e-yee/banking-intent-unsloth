import json
import polars as pl

from pathlib import Path
from datasets import load_dataset, DatasetDict

from constants import TAGS

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "sample_data"

def download_dataset(
    path: str, 
    revision: str
) -> tuple:
    """Download dataset with revision and convert to polars.DataFrames."""
    
    print(f"{TAGS["info"]} Downloading dataset from {path} (revision={revision})...")
    
    dataset: DatasetDict = load_dataset(
        path=path,
        revision=revision
    )
    
    output_path = DATA_DIR / "raw"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Converts to Polars
    train_pl = pl.from_arrow(dataset["train"].data.table)
    print(f"{TAGS["info"]} Loaded {train_pl.shape[0]:,} rows (train data).")
    
    test_pl = pl.from_arrow(dataset["test"].data.table)
    print(f"{TAGS["info"]} Loaded {test_pl.shape[0]:,} rows (test data).")
    
    # Labels
    label_feature = dataset["train"].features.get("label")
    
    labels = None
    if label_feature:
        labels = label_feature.names
        print(f"{TAGS["info"]} Loaded {len(labels)} labels.")
        
        categories_path = output_path / "categories.json"
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
            
        print(f"{TAGS["info"]} Saved labels to {categories_path}.")
    else:
        print(f"{TAGS["warning"]} No labels found.")
    
    # Save dataset
    print(f"{TAGS["info"]} Saving dataset to {output_path}...")
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_pl.write_csv(train_path)
    test_pl.write_csv(test_path)
    
    print(f"{TAGS["success"]} Dataset saved successfully.")
    
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
    
    label_map = {i:label for i, label in enumerate(labels)}
    
    def clean_data(data: pl.DataFrame) -> pl.DataFrame:
        return (
            data
            .with_columns(
                pl.col("text")
                .map_elements(clean_text, return_dtype=pl.String)
                .alias("cleaned_text"),
                
                pl.col("label")
                .replace_strict(label_map, return_dtype=pl.String)
                .alias("label_name")
            )
            .select(["cleaned_text", "label_name"])
        )
    
    # Clean data
    print(f"{TAGS["info"]} Cleaning dataset...")
    
    cleaned_train_data = clean_data(train_data)
    cleaned_test_data = clean_data(test_data)
    
    # Save data
    output_path = DATA_DIR / "preprocessed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{TAGS["info"]} Saving preprocessed dataset to {output_path}...")
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    cleaned_train_data.write_csv(train_path)
    cleaned_test_data.write_csv(test_path)
    
    with open(output_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    print(f"{TAGS["success"]} Preprocessed dataset saved successfully.")
    
    return cleaned_train_data, cleaned_test_data, label_map

def main():
    train_pl, test_pl, labels = download_dataset("PolyAI/banking77", "refs/convert/parquet")
    preprocess_data(train_pl, test_pl, labels)
    
if __name__ == "__main__":
    main()