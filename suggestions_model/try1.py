from llama_cpp import Llama

# llm = Llama(
#     model_path=r"C:\Users\stijn\ownCloud\DSP\models\geitje-7b-ultra-q8_0.gguf",
#     n_gpu_layers=30,
#     n_ctx=32768,
#     chat_format="chatml",
#     verbose=False,
# )

llm = Llama.from_pretrained(
    repo_id="mradermacher/GEITje-7B-ultra-i1-GGUF",
    filename="GEITje-7B-ultra.i1-Q4_K_M.gguf",
)

text = input()

messages = [
    {
        "role": "system",
        "content": """
            Jij bent een expert in de Nederlandse taal en geeft alleen antwoord in json format zonder extra uitleg erbij.
            Je weet dat Taalniveau B1 het normale taalniveau is in Nederland. Wanneer je taalniveau B1 beheerst, begrijp je teksten die grotendeels bestaan uit woorden en zinnen die je vaak hoort en waar je bekend mee bent. Zoals zaken die je vaak tegenkomt op je werk of school. Ook kan je je eigen mening geven en verantwoorden of een verhaal vertellen over bekende zaken en onderwerpen.
            Ook weet je dat het bij taalniveau B2 het je geen moeite om spontaan deel te nemen aan een gesprek. Ben je bekend met het onderwerp? Dan kan je meedoen aan een discussie en je standpunten duidelijk maken. Je begrijpt de kern van moeilijke teksten en kan zelf ook gedetailleerde teksten schrijven en spreken.
            Ook weet je dat het bij taalniveai C1 je lange en moeilijke teksten begrijpen. Zelfs als er onduidelijke verbindingen worden gelegd. Ook literaire teksten begrijp je. Je kan jezelf makkelijk uitdrukken en gedetailleerde omschrijvingen geven over lastige onderwerpen. De taal zet je flexibel in op het werk en voor sociale doeleinden.
            Ook weet je dat je bij C2 elke tekst kan gebruiken.
        """,
    },
    {
        "role": "user",
        "content": f"""
        Bepaal het taalniveau van deze tekst en als het niveau moeilijker dan B1 is, geef suggesties zodat het taalniveau B1 wordt:

        {text}
        """,
    },
]

output = llm.create_chat_completion(
    messages=messages,
    temperature=0.4,
    # response_format={
    #     "type": "json_object",
    #     "schema": {
    #         "type": "object",
    #         "properties": {
    #             "taalniveau": {"type": "string"},
    #             "suggesties": {
    #                 "type": ["object", "null"],  # Can be an object or null
    #                 "properties": {
    #                     "type": "object",
    #                     "additionalProperties": {"type": "string"}  # Keys and values are strings
    #                 }
    #             }
    #         },
    #         "required": ["taalniveau"]  # Only taalniveau is required
    #     }
    # }
)

print(output)
