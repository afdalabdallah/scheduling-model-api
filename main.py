import os

from flask import Flask, request, jsonify
import GA
import requests
app = Flask(__name__)

app.config['TIMEOUT'] = 30

@app.route('/', methods=['GET'])
def index():
    return "API is Online"

@app.route("/schedule", methods=['POST'])
def generate_schedule():
    test = request.json
    object = GA.GeneticAlgorithm(10,test)
    result = object.run()
    response = requests.post('http://localhost:5000/jadwal', json=result)
    response_data = {}
    if response:
        response_data = {"status": "success"}
    else:
        response_data = {"status": "Not success at generating"}

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))