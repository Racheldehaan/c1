import pandas as pd
from datasets import Dataset, load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import torch

# Step 1: Load the CSV dataset
df = pd.read_csv("language_levels.csv")

# Step 2: Convert the CSV to Hugging Face dataset format
dataset = Dataset.from_pandas(df)

# Step 3: Split the dataset into train and test sets
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
test_dataset = train_test['test']

# Step 4: Load a SetFit model from Hub or a pre-trained model
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",  # You can change this to a model that works well with your task
    labels=["A1", "A2", "B1", "B2", "C1", "C2"]
)

# device = torch.device("cpu")  # Force using CPU
# model.to(device)  # Move the model to the CPU

# Step 5: Format your dataset
# Ensure that your dataset contains 'text' and 'label' columns as expected by SetFit
train_dataset = train_dataset.rename_column("text", "sentence")
# train_dataset = train_dataset.rename_column("level_column_in_your_dataset", "label")

test_dataset = test_dataset.rename_column("text", "sentence")
# test_dataset = test_dataset.rename_column("level_column_in_your_dataset", "label")

# Step 6: Sample the dataset (for few-shot learning) - Optional step for faster experimentation
train_dataset = sample_dataset(train_dataset, label_column="label", num_samples=8)

# Step 7: Define the TrainingArguments for the Trainer
args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Step 8: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric="accuracy",  # Can be adjusted based on what metric you care about
    column_mapping={"sentence": "text", "label": "label"}  # Ensure mapping is correct
)

# Step 9: Train the model
trainer.train()

# Step 10: Evaluate the model
metrics = trainer.evaluate(test_dataset)
print(metrics)

# Step 11: Save the model after training (optional)
trainer.save_model("language_level_model")

# Step 12: Make predictions using the trained model
def predict_language_level(texts):
    preds = model.predict(texts)
    return preds

# Example texts to predict
texts_to_predict = [
    "Ik woon in Nederland. Mijn naam is Anna. Ik heb een hond. Ik ga elke dag wandelen in het park.",
    "De economie is een belangrijk onderwerp in de politiek. Veel mensen maken zich zorgen over de toekomst van de arbeidsmarkt. Er wordt gesproken over de gevolgen van automatisering en digitalisering. We moeten manieren vinden om werkgelegenheid te behouden in deze snel veranderende wereld."
]

predicted_levels = predict_language_level(texts_to_predict)

# Print the predictions
for text, level in zip(texts_to_predict, predicted_levels):
    print(f"Text: {text}\nPredicted Language Level: {level}\n")
