from flask import Flask, jsonify, request

from HeadOscillation import Head
from EyeDetection import ADHD_Calculation
from SpeechSpeed import SpeechSpeedClass

app = Flask(__name__)


# Route to get a list of items
# @app.route('/api/items', methods=['GET'])
# def get_items():
#     return jsonify({"items": items})


# Route to create a new item (POST request)
@app.route('/speech_speech', methods=['POST'])
def create_item():
    req_body = request.json  # Assuming the client sends JSON data with the request
    if not req_body:
        return jsonify({"error": "Data not provided in JSON format"}), 400
    
    # Speech to text process
    speech_class = SpeechSpeedClass(req_body["audio_path"])
    speech_class.calculate_speaking_speed()    

    return jsonify({"message": "Item created successfully", "item": req_body}), 201


@app.route('/head_score', method = ['POST'])
def get_head_score():
    req_body = request.json
    

if __name__ == '__main__':
    app.run(debug=True)
