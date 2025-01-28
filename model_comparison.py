import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import torch
from evaluate import load
import numpy as np
import random

# Set seeds for reproducibility
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration
MODELS_TO_COMPARE = [
    {
        "name": "multilingual-miniLM",
        "pretrained": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    },
    {
        "name": "dutch-robbert",
        "pretrained": "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    }
]

# Initialize metrics
METRICS = {
    "accuracy": load("accuracy"),
    "precision": load("precision"),
    "recall": load("recall"),
    "f1": load("f1"),
}

label_to_num = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

def is_checkmark(label):
    return label <= label_to_num["B1"]

def train_and_evaluate_model(model_config, train_dataset, test_dataset):
    print(f"\nTraining {model_config['name']}...")

    # Initialize model
    model = SetFitModel.from_pretrained(
        model_config["pretrained"],
        labels=["A1", "A2", "B1", "B2", "C1", "C2"]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Training arguments (match original)
    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer with correct column mapping
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        column_mapping={"sentence": "text", "label": "label"},
    )

    # Train
    trainer.train()

    # Evaluate and compute metrics
    predictions = model.predict(test_dataset["sentence"])
    y_true = [label_to_num[label] for label in test_dataset["label"]]
    y_pred = [label_to_num[label] for label in predictions]

    # Multiclass metrics
    accuracy = METRICS["accuracy"].compute(predictions=y_pred, references=y_true)["accuracy"]
    precision = METRICS["precision"].compute(predictions=y_pred, references=y_true, average="weighted")["precision"]
    recall = METRICS["recall"].compute(predictions=y_pred, references=y_true, average="weighted")["recall"]
    f1 = METRICS["f1"].compute(predictions=y_pred, references=y_true, average="weighted")["f1"]
    mae = float(np.mean(np.abs(np.array(y_pred) - np.array(y_true))))

    # Checkmark metrics
    checkmark_true = [int(is_checkmark(label)) for label in y_true]
    checkmark_pred = [int(is_checkmark(label)) for label in y_pred]

    checkmark_accuracy = METRICS["accuracy"].compute(predictions=checkmark_pred, references=checkmark_true)["accuracy"]
    checkmark_precision = METRICS["precision"].compute(predictions=checkmark_pred, references=checkmark_true, average="binary", pos_label=1)["precision"]
    checkmark_recall = METRICS["recall"].compute(predictions=checkmark_pred, references=checkmark_true, average="binary", pos_label=1)["recall"]
    checkmark_f1 = METRICS["f1"].compute(predictions=checkmark_pred, references=checkmark_true, average="binary", pos_label=1)["f1"]

    return {
        "model": model_config["name"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "checkmark_accuracy": checkmark_accuracy,
        "checkmark_precision": checkmark_precision,
        "checkmark_recall": checkmark_recall,
        "checkmark_f1": checkmark_f1,
    }

def compare_models():
    # Load and prepare data
    df = pd.read_csv("language_levels.csv")
    df['label'] = df['label'].str.strip()

    # Convert to dataset and split
    dataset = Dataset.from_pandas(df)
    train_test = dataset.train_test_split(test_size=0.35)  # No seed to match original

    # Rename columns to match original preprocessing
    train_dataset = train_test["train"].rename_column("text", "sentence")
    test_dataset = train_test["test"].rename_column("text", "sentence")

    # Compare models
    comparison_results = []
    for model_config in MODELS_TO_COMPARE:
        result = train_and_evaluate_model(model_config, train_dataset, test_dataset)
        comparison_results.append(result)

    # Display results
    results_df = pd.DataFrame(comparison_results).set_index("model")
    print("\nModel Comparison Results:")
    print(results_df)

    return results_df

if __name__ == "__main__":
    results = compare_models()