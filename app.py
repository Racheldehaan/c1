from flask import Flask, render_template, request
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import string

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['DEBUG'] = True

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Login to Hugging Face
login("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")

# Model ID
model_id = "BramVanroy/GEITje-7B-ultra"

# Directory to save/load the model and tokenizer
cache_dir = "./model_cache"

# Load or download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir)

# Save model and tokenizer locally for future reuse
tokenizer.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cpu",
    model_kwargs={"torch_dtype": torch.float16},
    max_length=200,
)

messages = [
        # {"role": "system", "content": """Jij bent een expert in de Nederlandse taal, je krijgt een stukje text waarvan je streng het taalnivea moet bepalen. Zeg alleen welk niveau het is, dus bijvoorbeeld B1 of C1. Zeg niet waarom het zo is, alleen het niveau.
        # Taalniveau B1 (normaal)
        # Taalniveau B1 is het normale taalniveau in Nederland. Wanneer je taalniveau B1 beheerst, begrijp je teksten die grotendeels bestaan uit woorden en zinnen die je vaak hoort en waar je bekend mee bent. Zoals zaken die je vaak tegenkomt op je werk of school. Ook kan je je eigen mening geven en verantwoorden of een verhaal vertellen over bekende zaken en onderwerpen.
        # """
        # },
        {"role": "system", "content": """
        Bepaal het taalniveau van deze tekst en als het niveau moeilijker dan B1 is, geef suggesties zodat het taalniveau B1 wordt:

        Begin text:
        {text}
        Einde text.

        Geef het antwoord in het volgende formaat:
        { "taalniveau" : ¨-vul in taalniveau, gebruik maximaal 2 characters-¨,
        ¨suggesties" : None als er geen suggesties zijn, anders: { 
            "-originele zin-": "-vereenvoudigde zin-", 
            "-andere originele zin-": "-andere vereenvoudigde zin-" 
            }
        }

        Voorbeeld output:
        {
        "taalniveau": "B2",
        "suggesties": {
        "-De oude man liep langzaam door de straat, zijn rug gebogen van de jaren-": "-De oude man liep langzaam, zijn rug gebogen van de jaren.",
        "-De minister verklaarde dat er aanzienlijke veranderingen nodig zijn om de situatie te verbeteren-": "-De minister zei dat er veel veranderingen nodig zijn om de situatie beter te maken."
        }
        }
        """}
    ]

@app.route('/', methods=['GET'])
def home():
    return render_template('base.html')

# @app.route('/output', methods=['POST'])
# def output():
#     text_input = request.form.get('text-input')
#     suggestions = []
    
#     # Hier komt de logica voor het maken van suggesties op basis van de input
#     if text_input:
#         suggestions = [f"• {x}" for x in text_input]
#         language_level = "B2"
    
#     else:
#         language_level = None
    
#     return render_template('base.html', suggestions=suggestions, language_level=language_level)


@app.route('/output', methods=['POST'])
def output():
    text_input = request.form.get('text-input')
    suggestions = []
    language_level = None

    if text_input:
        # Update user message with the input text
        
        user_message = messages[0].copy()
        user_message["content"] = user_message["content"].replace("{text}", text_input)
        messages[0] = user_message

        # Get language level from model response
        pipe.generation_config.pad_token_id = tokenizer.eos_token_id
        outputs = pipe(messages, max_new_tokens=200)
        language_level = outputs[0]['generated_text'][-1]['content'].strip()

        # TO DO: suggestions
        suggestions = [f"• {x}" for x in text_input]

    return render_template('base.html', suggestions=suggestions, language_level=language_level)


if __name__ == "__main__":
    app.run(debug=True)
