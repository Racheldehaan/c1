import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import torch
from evaluate import load
import numpy as np
import random

# Set seeds
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
        "pretrained": "pdelobelle/robbert-v2-dutch-base"
    }
]

# Initialize metrics
METRICS = {
    "accuracy": load("accuracy"),
    "precision": load("precision", average="weighted"),
    "recall": load("recall", average="weighted"),
    "f1": load("f1", average="weighted"),
}

def train_and_evaluate_model(model_config, train_dataset, test_dataset):
    """Train and evaluate a single model configuration"""
    print(f"\nTraining {model_config['name']}...")

    # Initialize model
    model = SetFitModel.from_pretrained(
        model_config["pretrained"],
        labels=["A1", "A2", "B1", "B2", "C1", "C2"]
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Training arguments
    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        body_learning_rate=2e-5,  # For the base transformer model
        head_learning_rate=1e-3,  # For the classification head
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer with corrected column mapping
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        column_mapping={"text": "text", "label": "label"},  # Adjusted mapping
    )

    # Training
    trainer.train()

    # Evaluation
    results = trainer.evaluate()

    # Calculate MAE manually
    label_to_num = {label: i for i, label in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"])}
    predictions = model.predict(test_dataset["text"])
    y_true = [label_to_num[label] for label in test_dataset["label"]]
    y_pred = [label_to_num[label] for label in predictions]

    mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

    # Calculate checkmark metrics
    checkmark_true = [int(label <= label_to_num["B1"]) for label in y_true]
    checkmark_pred = [int(label <= label_to_num["B1"]) for label in y_pred]

    return {
        "model": model_config["name"],
        "multiclass_accuracy": results["accuracy"],
        "mae": mae,  # Use our manually calculated MAE
        "checkmark_accuracy": METRICS["accuracy"].compute(
            predictions=checkmark_pred,
            references=checkmark_true
        )["accuracy"],
        "checkmark_precision": METRICS["precision"].compute(
            predictions=checkmark_pred,
            references=checkmark_true,
            average="binary"
        )["precision"],
        "checkmark_recall": METRICS["recall"].compute(
            predictions=checkmark_pred,
            references=checkmark_true,
            average="binary"
        )["recall"],
    }

def compare_models():
    # Load and prepare data
    df = pd.read_csv("language_levels.csv")
    df['label'] = df['label'].str.strip()

    # Use original column names
    dataset = Dataset.from_pandas(df.rename(columns={
        "text": "text",  # Keep original column name
        "label": "label"
    }))

    train_test = dataset.train_test_split(test_size=0.35, seed=SEED)

    # Compare models
    comparison_results = []
    for model_config in MODELS_TO_COMPARE:
        result = train_and_evaluate_model(
            model_config,
            train_test["train"],
            train_test["test"]
        )
        comparison_results.append(result)

    # Display results
    results_df = pd.DataFrame(comparison_results).set_index("model")
    print("\nModel Comparison Results:")
    print(results_df)

    return results_df

if __name__ == "__main__":
    results = compare_models()