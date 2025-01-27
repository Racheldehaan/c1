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
    filename="GEITje-7B-ultra.i1-Q4_K_M.gguf",
    # n_threads_batch=8,
    n_ctx=4096,  # Reduce from 32768
    n_gpu_layers=43,  # Start with 30 layers for 8GB VRAM
    n_threads=8,  # Match your 8-core CPU
    n_batch=256,  # Optimal for GPU processing
    offload_kqv=False,  # Offload attention layers to GPU
    flash_attn=True
)

# llm.verbose = False  # For not printing model information

# llm = Llama.from_pretrained(
#     repo_id="mradermacher/GEITje-7B-ultra-i1-GGUF",
#     filename="GEITje-7B-ultra.i1-Q4_K_M.gguf",
#     n_threads_batch=8,
#     n_ctx=4096,  # Reduce from 32768
#     n_gpu_layers=30,  # Start with 30 layers for 8GB VRAM
#     n_threads=8,  # Match your 8-core CPU
#     n_batch=512,  # Optimal for GPU processing
#     offload_kqv=True,  # Offload attention layers to GPU
#     flash_attn=True
# )

messages = [
    {
        "role": "system",
        "content": """
            Jij bent een juridisch taalexpert. Vereenvoudig juridische teksten naar B1-niveau MET BEHOUD VAN:
            1. Originele aanspreekvorm (u/uw blijft u/uw)
            2. Exacte volgorde van zinnen en alinea's
            3. Alle juridische verwijzingen en artikelnummers
            4. Specifieke feiten en claims
            5. Formele toon

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
        Vereenvoudig onderstaande juridische tekst naar B1-niveau. Behoud alle juridische verwijzingen en artikelnummers en zorg dat het formeel blijft.
        Voeg waar nodig korte verduidelijkingen toe tussen haakjes. Splits zinnen niet.

        Tekst:
        {text}
        """
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

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    completion = llm.create_chat_completion(
                        messages=local_messages,
                        temperature=0.5,
                        repeat_penalty=1.1,
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
                        raise ValueError("Invalid response format from LLM model")

                except Exception as e:
                    # Log the error (could be to a file, monitoring system, etc.)
                    print(f"Attempt {attempt + 1} failed: {str(e)}")

                    # If the maximum retries have been reached, return an error
                    if attempt == max_retries - 1:
                        return f"Error processing LLM response after {max_retries} attempts: {str(e)}", 500

        elif language_level in ["A1", "A2", "B1"]:
            return render_template(
                "base.html", suggestions=[], language_level=language_level
            )

        else:
            return "Error: Invalid language level detected", 500

    return render_template("base.html", suggestions=[], language_level=None)



if __name__ == "__main__":
    app.run(debug=True)
