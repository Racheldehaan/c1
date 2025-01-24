import torch
import string
import os
import pandas as pd
from flask import Flask, render_template, request
import json

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import Dataset, load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from llama_cpp import Llama

app = Flask(
    __name__, template_folder=os.path.join(os.path.dirname(__file__), "templates")
)

app.config["DEBUG"] = True

# Load the saved model
model = SetFitModel.from_pretrained("language_level_model")

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)
print(f"Using device: {device}")

def predict_language_level(texts):
    preds = model.predict(texts)
    return preds


llm = Llama.from_pretrained(
    repo_id="mradermacher/GEITje-7B-ultra-i1-GGUF",
    filename="GEITje-7B-ultra.i1-Q5_K_M.gguf",
    n_ctx=2048*4,
    n_gpu_layers=-1,
)

messages = [
    {
        "role": "system",
        "content": """
            Jij bent een expert in de Nederlandse taal en geeft alleen antwoord in json format zonder extra uitleg erbij. Dit is wat je weet over taalniveaus in het Nederlands op volgorde van makkelijk naar moeilijk plus een uitleg:
            A1: Iemand die (alleen) taalniveau A1 begrijpt en spreekt, is een beginnend taalgebruiker. Hij begrijpt eenvoudige woorden en namen en heel korte zinnen.
            A2: A2 is ook heel eenvoudig. Maar de zinnen zijn al iets langer. Iemand met dit niveau begrijpt de boodschap van korte, eenvoudige teksten. Die teksten moeten duidelijk zijn en gaan over de eigen omgeving. Als je schrijft voor laaggeletterden, is dit niveau geschikt.
            B1: Het niveau dat de meeste Nederlanders begrijpen. B1 draait om eenvoudige en duidelijke taal. Mensen met dit taalniveau begrijpen de meeste teksten die over veelvoorkomende onderwerpen gaan. Het lijkt een beetje op spreektaal. Een van de kenmerken van taalniveau B1 en de onderliggende niveaus, is een duidelijke tekststructuur.
            B2: Iemand die taalniveau B2 begrijpt, snapt ingewikkeldere teksten. Al helemaal als het gaat over een (wat moeilijker) onderwerp dat hij in zijn eigen beroep of interessegebied tegenkomt.
            C1: Heeft iemand taalniveau C1, dan begrijpt hij moeilijke, lange teksten, ook als die abstract (vaag) zijn. Hij begrijpt vaktaal, uitdrukkingen, ouderwetse woorden en moeilijke woorden. En hij kan taal zelf goed inzetten om iets uit te leggen.
            C2: Dit is de moeilijkste van alle taalniveaus. Iemand die C2 begrijpt, begrijpt eigenlijk alles wat in het Nederlands wordt gezegd of geschreven.

            Geef het antwoord in het volgende formaat:
            {"suggesties" : None als er geen suggesties zijn, anders een lijst met minimaal lengte 1 en geen maximale lengte met de suggesties in het volgende formaat: [
              {
              "zin uit de tekst die aangepast moet worden": "-de zin-",
              "wat het zou moeten worden": "-vereenvoudigde zin-"
              },
              {
              "zin uit de tekst die aangepast moet worden": "-de zin-",
              "wat het zou moeten worden": "-vereenvoudigde zin-"
              },
              {
              "zin uit de tekst die aangepast moet worden": "-de zin-",
              "wat het zou moeten worden": "-vereenvoudigde zin-"
              },
              ....


            }
        """,
    },
    {
        "role": "user",
        "content": """
        Deze tekst heeft een moeilijker taalniveau dan B1, geef suggesties zodat het taalniveau B1 wordt:

        {text}
        """,
    },
]


@app.route("/", methods=["GET"])
def home():
    return render_template("base.html")


@app.route("/output", methods=["POST"])
def output():
    text_input = request.form.get("text-input")

    if text_input:
        predicted_levels = predict_language_level([text_input])
        language_level = predicted_levels[0]

        if language_level in ["B2", "C1", "C2"]:
            local_messages = [
                {
                    "role": msg["role"],
                    "content": (
                        msg["content"].replace("{text}", text_input)
                        if "{text}" in msg["content"]
                        else msg["content"]
                    ),
                }
                for msg in messages
            ]

            try:
                completion = llm.create_chat_completion(
                    messages=local_messages,
                    temperature=0.4,
                    response_format={
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "suggesties": {
                                    "type": ["array", "null"],
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "zin uit de tekst die aangepast moet worden": {
                                                "type": "string"
                                            },
                                            "wat het zou moeten worden": {
                                                "type": "string"
                                            },
                                        },
                                        "required": [
                                            "zin uit de tekst die aangepast moet worden",
                                            "wat het zou moeten worden",
                                        ],
                                    },
                                },
                            },
                        },
                    },
                )

                suggest = json.loads(completion["choices"][0]["message"]["content"])

                if isinstance(suggest.get("suggesties"), list):
                    formatted_suggestions = [
                        {
                            "id": i,
                            "original": item["zin uit de tekst die aangepast moet worden"],
                            "new": item["wat het zou moeten worden"],
                        }
                        for i, item in enumerate(suggest["suggesties"], start=1)
                    ]

                    return render_template(
                        "base.html",
                        suggestions=formatted_suggestions,
                        language_level=language_level,
                    )
                else:
                    return "Error: Invalid response format from LLM model", 500

            except Exception as e:
                return f"Error processing LLM response: {str(e)}", 500

        elif language_level in ["A1", "A2", "B1"]:
            return render_template(
                "base.html", suggestions=[], language_level=language_level
            )

        else:
            return f"Error processing LLM response: {str(e)}", 500

    return render_template("base.html", suggestions=[], language_level=None)


if __name__ == "__main__":
    app.run(debug=True)
