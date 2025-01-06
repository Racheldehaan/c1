from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def home():
    return render_template('base.html')

@app.route('/output', methods=['POST'])
def output():
    text_input = request.form.get('text-input')
    suggestions = []
    
    # Hier komt de logica voor het maken van suggesties op basis van de input
    if text_input:
        suggestions = [f"• {x}" for x in text_input]
        language_level = "B2"
    
    else:
        language_level = None
    
    return render_template('base.html', suggestions=suggestions, language_level=language_level)

if __name__ == "__main__":
    app.run(debug=True)
