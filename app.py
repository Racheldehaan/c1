from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['DEBUG'] = True 

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('base.html')

if __name__ == "__main__":
    app.run(debug=True)