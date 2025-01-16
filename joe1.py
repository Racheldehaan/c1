from llama_cpp import Llama

# llm = Llama(
#     model_path=r"C:\Users\stijn\ownCloud\DSP\models\GEITje-7B-ultra.i1-Q4_K_M.gguf",
#     n_gpu_layers=0,
#     n_ctx=32768,
#     chat_format="chatml",
#     verbose=False,
# )

# llm = Llama.from_pretrained(
# 	repo_id="mradermacher/GEITje-7B-ultra-i1-GGUF",
# 	filename="GEITje-7B-ultra.i1-Q4_K_M.gguf",
#     n_ctx=32768
# )

llm = Llama.from_pretrained(
	repo_id="mradermacher/GEITje-7B-ultra-i1-GGUF",
	filename="GEITje-7B-ultra.i1-Q5_K_M.gguf",
    n_ctx=32768
)



text = input()

messages = [
    {
        "role": "system",
        "content": """
            Jij bent een expert in de Nederlandse taal en geeft alleen antwoord in json format zonder extra uitleg erbij. Dit zijn de mogelijke taalniveaus in het Nederlands die jij op volgorde van makkelijk naar moeilijk plus een uitleg:
            A1: Iemand die (alleen) taalniveau A1 begrijpt en spreekt, is een beginnend taalgebruiker. Hij begrijpt eenvoudige woorden en namen en heel korte zinnen.
            A2: A2 is ook heel eenvoudig. Maar de zinnen zijn al iets langer. Iemand met dit niveau begrijpt de boodschap van korte, eenvoudige teksten. Die teksten moeten duidelijk zijn en gaan over de eigen omgeving. Als je schrijft voor laaggeletterden, is dit niveau geschikt.
            B1: Het niveau dat de meeste Nederlanders begrijpen. B1 draait om eenvoudige en duidelijke taal. Mensen met dit taalniveau begrijpen de meeste teksten die over veelvoorkomende onderwerpen gaan. Het lijkt een beetje op spreektaal. Een van de kenmerken van taalniveau B1 en de onderliggende niveaus, is een duidelijke tekststructuur.
            B2: Iemand die taalniveau B2 begrijpt, snapt ingewikkeldere teksten. Al helemaal als het gaat over een (wat moeilijker) onderwerp dat hij in zijn eigen beroep of interessegebied tegenkomt.
            C1: Heeft iemand taalniveau C1, dan begrijpt hij moeilijke, lange teksten, ook als die abstract (vaag) zijn. Hij begrijpt vaktaal, uitdrukkingen, ouderwetse woorden en moeilijke woorden. En hij kan taal zelf goed inzetten om iets uit te leggen.
            C2: Dit is de moeilijkste van alle taalniveaus. Iemand die C2 begrijpt, begrijpt eigenlijk alles wat in het Nederlands wordt gezegd of geschreven.

            Geef het antwoord in het volgende formaat:
            { "taalniveau" : "",
              "suggesties" : None als er geen suggesties zijn, anders een lijst met minimaal lengte 1 en geen maximale lengte met de suggesties in het volgende formaat: [
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
        """
    },

    {"role": "user", "content": f"""
        Bepaal het taalniveau van de hierop volgende tekst en als het niveau moeilijker dan B1 is, geef suggesties zodat het taalniveau B1 wordt:

        {text}
        """}
]

output = llm.create_chat_completion(
    messages=messages,
    temperature=0.4,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
            "taalniveau": {
                "type": "string",
                "maxLength": 2 
            },
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
                    }
                },
                "required": ["zin uit de tekst die aangepast moet worden", "wat het zou moeten worden"]
                }
            }
            },
            "required": ["taalniveau"]
        }
        }

)

print(output)
