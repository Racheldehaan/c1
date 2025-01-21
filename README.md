# Language Level Tool

This tool can be used to check whether the language level of a Dutch text is easy enough for most people to understand. Most people understand a B1 level, if the text is more complicated, suggestions are given to simplify the text. It is possible to either upload a file from which it will extract the text or to write the text within the application.

## Set up
### Installing requirements
Run the following to install all requirements for this tool:
```
pip install -r requirements.txt
```

### The models
To download the model that does the classification of the text, you need to run:
```
python AI_model_classification.py
```
This file will download a text classification model and fine tune it on this specific use case. It uses the examples found in `language_levels.csv.'. If you want the model to be trained on your specific type of text, you can add the texts to this csv file.

Another model handles creating the suggestions when the text is too complicated. This model is automatically downloaded when the application is started for the first time and does not need to be fine-tuned.

## Start the Tool
To start the tool locally, run:
```
flask --app app run
```

This starts a flask app that you can open in your browser. You can now fill in your text or upload a file to check the language level and to receive suggestions if the text is too difficult. 

## Credits
This project was done as part of the Data Systems Project from the UvA for the municipality of Amsterdam by Stijn Lakeman, Davy Paardenkooper, Jippe van Roon and Rachel de Haan.