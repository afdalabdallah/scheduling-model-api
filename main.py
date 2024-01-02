import os

from flask import Flask, request, jsonify
import GA

app = Flask(__name__)

app.config['TIMEOUT'] = 30

@app.route('/', methods=['GET'])
def index():
    return "API is Online"

@app.route("/schedule", methods=['POST'])
def generate_schedule():
    test = request.json
    object = GA.GeneticAlgorithm(10,test)
    return object.run()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))