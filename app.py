import json
from flask import Flask, render_template, session, redirect, url_for, session, request
import shakespeare

app = Flask(__name__)


@app.route('/shakespeare', methods=['GET'])
def index1():
    try:
        text = request.args.get("text")
        characters = request.args.get("characters")
        characters = int(characters)
        generated_text = shakespeare.generate_text_2(text, characters)
        data_set = {"generated_text": generated_text, "text": text, "characters": characters, "status": "200OK"}
        json_dump = json.dumps(data_set)
    except:
        data_set = {"generated_text": "ERROR", "status": "ERROR"}
        json_dump = json.dumps(data_set)
    return json_dump


@app.route('/')
def index3():
    return "Welcome to this Api System"


if __name__ == '__main__':
    app.run(debug=True)
