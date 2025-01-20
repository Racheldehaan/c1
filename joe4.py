import pandas as pd
from datasets import Dataset, load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import torch
from evaluate import load
import numpy as np

# Step 1: Load the CSV dataset
df = pd.read_csv("language_levels.csv")

# Step 2: Convert the CSV to Hugging Face dataset format
dataset = Dataset.from_pandas(df)

# Step 3: Split the dataset into train and test sets
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

# Step 4: Load a SetFit model from Hub or a pre-trained model
# You can also try any other model from: https://huggingface.co/models?library=sentence-transformers&language=nl&sort=downloads
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # You can change this to a model that works well with your task
    labels=["A1", "A2", "B1", "B2", "C1", "C2"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Force using CPU
model.to(device)  # Move the model to the CPU

# Step 5: Format your dataset
# Ensure that your dataset contains 'text' and 'label' columns as expected by SetFit
train_dataset = train_dataset.rename_column("text", "sentence")
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


# Step 8: Initialize the metrics (accuracy, precision, recall, f1)
accuracy_metric = load("accuracy")
precision_metric = load("precision", average="weighted")
recall_metric = load("recall", average="weighted")
f1_metric = load("f1", average="weighted")


def is_lower_or_higher_than_B1(label):
    label_to_num = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    return label < label_to_num["B1"]

def custom_metric(y_pred, y_test):
    accuracy = accuracy_metric.compute(predictions=y_pred, references=y_test)
    precision = precision_metric.compute(predictions=y_pred, references=y_test, average="weighted")
    recall = recall_metric.compute(predictions=y_pred, references=y_test, average="weighted")
    f1 = f1_metric.compute(predictions=y_pred, references=y_test, average="weighted")
    mae = np.mean(np.abs(np.array(y_pred) - np.array(y_test)))

    # Calculate the accuracy of predicting if the level is lower or higher than B1
    y_test_binary = [is_lower_or_higher_than_B1(label) for label in y_test]
    y_pred_binary = [is_lower_or_higher_than_B1(label) for label in y_pred]
    binary_accuracy = accuracy_metric.compute(predictions=y_pred_binary, references=y_test_binary)

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
        "mae": mae,
        "binary_accuracy": binary_accuracy["accuracy"],
    }


# Step 9: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric=custom_metric,  # Can be adjusted based on what metric you care about
    column_mapping={"sentence": "text", "label": "label"},  # Ensure mapping is correct
)

# Step 10: Train the model
trainer.train()

# Step 11: Evaluate the model
metrics = trainer.evaluate(test_dataset)
print(metrics)

# Step 12: Save the model after training (optional)
# trainer.save_model("language_level_model")
# Save the trained model
model.save_pretrained("language_level_model")


# Step 13: Make predictions using the trained model
def predict_language_level(texts):
    preds = model.predict(texts)
    return preds


# Example texts to predict
texts_to_predict = [
    # A1 - Zeer eenvoudig
    "Ik woon in Nederland. Mijn naam is Anna. Ik heb een hond. Ik ga elke dag wandelen in het park.",
    "Dit is een appel. Ik eet de appel. Het is lekker. De appel is rood.",
    # A2 - Iets langere en complexere zinnen
    "Mijn moeder is thuis. Zij werkt in een ziekenhuis. Ze zorgt voor patiënten die ziek zijn. Elke dag werkt ze hard om mensen beter te maken.",
    "Ik ga naar de winkel. Ik koop brood, melk en kaas. Daarna ga ik naar huis. Ik ben blij omdat ik lekker kan eten.",
    # B1 - Eenvoudige, maar langere en meer beschrijvende teksten
    "Ik woon in een klein dorp. Het is rustig hier en er is veel groen. In de zomer ga ik vaak fietsen in de natuur. Er zijn veel wandelroutes en ik geniet altijd van de frisse lucht.",
    "De stad is heel druk, maar ook interessant. Er zijn veel winkels, cafés en restaurants. In het weekend ga ik graag naar een café om koffie te drinken met vrienden. We praten over van alles, van werk tot vakantieplannen.",
    # B2 - Iets moeilijkere zinnen, uitleg en argumentatie
    "Technologie verandert snel. Elke dag worden er nieuwe apparaten en apps ontwikkeld. Dit heeft zowel voordelen als nadelen. Aan de ene kant kunnen we sneller communiceren en informatie vinden, maar aan de andere kant kunnen we ook afhankelijk worden van technologie. Het is belangrijk om een balans te vinden.",
    "De klimaatverandering is een groot probleem dat de hele wereld aangaat. De afgelopen jaren zijn er veel natuurrampen geweest, zoals overstromingen en bosbranden. Wetenschappers zeggen dat we ons gedrag moeten veranderen om de aarde te beschermen. Dit kan door minder energie te verbruiken en duurzamer te leven.",
    # C1 - Complexe teksten met gedetailleerde uitleg en abstracte thema's
    "Het is belangrijk om een kritische houding te hebben ten opzichte van de media. In een tijd van sociale media en 24/7 nieuws kunnen we gemakkelijk beïnvloed worden door nepnieuws en misinformatie. Het is essentieel om bronnen te verifiëren en te zorgen voor een evenwichtig perspectief. Alleen dan kunnen we weloverwogen beslissingen nemen over wat we geloven.",
    "De integratie van kunstmatige intelligentie in de gezondheidszorg biedt zowel kansen als uitdagingen. Aan de ene kant kunnen AI-systemen artsen helpen bij het stellen van diagnoses en het personaliseren van behandelingen. Aan de andere kant roept de toenemende afhankelijkheid van technologie ethische vragen op over privacy, autonomie en de rol van de mens in het zorgproces.",
    # C2 - Zeer complexe en diepgaande teksten, abstracte ideeën
    "De globalisering heeft de manier waarop we de wereld begrijpen fundamenteel veranderd. Terwijl economische barrières zijn afgebroken en handel tussen landen is toegenomen, heeft deze verandering ook geleid tot nieuwe sociaal-politieke uitdagingen. De kloof tussen rijke en arme landen is groter geworden, en er is een groeiende bezorgdheid over de gevolgen voor lokale culturen en de milieu-impact van wereldwijde productieprocessen. Het debat over de voor- en nadelen van globalisering blijft onverminderd intensief.",
    "De filosofie van existentiële onzekerheid onderzoekt de fundamentele vragen van het menselijk bestaan, zoals de betekenis van het leven, de rol van vrijheid en verantwoordelijkheid, en de ervaring van vervreemding. Filosoof Jean-Paul Sartre stelde dat mensen zelf hun essentie creëren door keuzes en daden, wat leidt tot de notie van 'authenticiteit'. Echter, deze vrijheid kan tegelijkertijd leiden tot existentiële angst, omdat het besef van verantwoordelijkheid een zware last is. Het zoeken naar betekenis in een ogenschijnlijk chaotische en zinloze wereld vormt de kern van het existentialisme.",
]


predicted_levels = predict_language_level(texts_to_predict)

# Print the predictions
for text, level in zip(texts_to_predict, predicted_levels):
    print(f"Text: {text}\nPredicted Language Level: {level}\n")
