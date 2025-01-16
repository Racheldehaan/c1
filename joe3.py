import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import torch

# Step 1: Load the CSV dataset
df = pd.read_csv("language_levels.csv")

# Step 2: Convert the CSV to Hugging Face dataset format
dataset = Dataset.from_pandas(df)

# Step 3: Split the dataset into train and test sets
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
test_dataset = train_test['test']

# Step 4: Load a pretrained model and tokenizer
model_name = "distilbert-base-uncased"  # You can replace this with a more suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)  # 6 language levels

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to the selected device

# Step 5: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Step 6: Format the dataset for PyTorch (it needs to have labels as integers)
label_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
train_dataset = train_dataset.map(lambda x: {'label': label_mapping[x['label']]})
test_dataset = test_dataset.map(lambda x: {'label': label_mapping[x['label']]})

# Step 7: Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluate every epoch
)

# Step 8: Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    tokenizer=tokenizer,                 # tokenizer
)

# Step 9: Train the model
trainer.train()

# Step 10: Save the model after training
trainer.save_model("language_level_model")

# Step 11: Make predictions using the trained model
def predict_language_level(texts):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)  # Move inputs to the same device as model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    # Convert the numerical predictions back to language levels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return [reverse_label_mapping[pred.item()] for pred in predictions]

# Example texts to predict
texts_to_predict = [
    "Ik woon in Nederland. Mijn naam is Anna. Ik heb een hond. Ik ga elke dag wandelen in het park.",
    "De economie is een belangrijk onderwerp in de politiek. Veel mensen maken zich zorgen over de toekomst van de arbeidsmarkt. Er wordt gesproken over de gevolgen van automatisering en digitalisering. We moeten manieren vinden om werkgelegenheid te behouden in deze snel veranderende wereld."
]

predicted_levels = predict_language_level(texts_to_predict)

# Print the predictions
for text, level in zip(texts_to_predict, predicted_levels):
    print(f"Text: {text}\nPredicted Language Level: {level}\n")
