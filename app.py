from flask import Flask, request, jsonify
import os 
import base64
from flask_cors import CORS
import numpy as np
import cv2
import json

import General

app = Flask(__name__)

# Define a dictionary to store data (in-memory database for this example)
data = {}

@app.route('/api/data', methods=['POST'])
def add_data():
    # Get the JSON data from the request
    request_data = request.get_json()

    # Check if the required fields are present
    if 'key' in request_data and 'value' in request_data:
        key = request_data['key']
        value = request_data['value']

        # Store the data in the dictionary
        data[key] = value

        # Return a response
        return jsonify({'message': f'Data with key "{key}" added successfully.'}), 201
    else:
        return jsonify({'message': 'Invalid request. Please provide both "key" and "value".'}), 400

def upload_audio(file):
    # Create a directory to store the uploaded files if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Get the file name
    filename = file.filename

    # Save the file to the 'uploads' directory
    file.save(os.path.join('uploads', filename))

    # Return the path to the uploaded file
    return os.path.join('uploads', filename)



@app.route('/ai/audio_temp', methods = ['POST'])
def get_audio_temp():
    print(request.files)
    uploaded_file = request.files['dosya.m4a']

    # Check if a file was uploaded
    if uploaded_file.filename != '':
        uploaded_path = upload_audio(uploaded_file)
    
        
        
        print("uploaded path:", uploaded_path)
        run_class.full_disorder_detection(audio_path=uploaded_path)

        return jsonify({"message": "işlem başarılı"})
    else:
        return jsonify({'error': 'No file uploaded'})
    
    
@app.route('/ai/predict_spiral_parkinson', methods = ['POST'])
def detect_spiral_parkinson():
    data = request.json  
    base64_image = data.get('image')
    
    # Decode the base64 image
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    run_class._spiral_parkinson_class.model_predict(img)
    
    
    return jsonify({'message': 'Parkinson Spiral process finished successfully',
                    "label": run_class._spiral_parkinson_class.estimated_label,
                    "ratio": run_class._spiral_parkinson_class.ratio})


@app.route('/ai/predict_wave_parkinson', methods = ['POST'])
def detect_wave_parkinson():
    data = request.json  
    base64_image = data.get('image')
    
    # Decode the base64 image
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    run_class._wave_parkinson_class.model_predict(img)
    
    
    return jsonify({'message': 'Parkinson Wave process finished successfully',
                    "label": run_class._wave_parkinson_class.estimated_label,
                    "ratio": run_class._wave_parkinson_class.ratio})
    
    
    
@app.route('/ai/image_temp', methods=['POST'])
def receive_frame():
    try:
        print(request.json)
        images = request.json['images']
        
        
            
        data = json_file.read()
        json_data = json.loads(data)

        # Access the images array
        images = json_data['images']

        # Process the images as needed
        for i, base64_data in enumerate(images):
            with open(f'uploaded_image_{i}.png', 'wb') as f:
                f.write(base64_data.decode('base64'))

        return jsonify({'message': 'JSON file uploaded successfully!'}), 200

    except Exception as e:
        return jsonify({'message': str(e)}), 500
if __name__ == '__main__':
    run_class = General.Run()
    app.run(debug=True, host='0.0.0.0', port=5000)
