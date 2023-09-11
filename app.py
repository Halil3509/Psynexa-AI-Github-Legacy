from flask import Flask, request, jsonify
import os 
import base64

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
    
    
    
@app.route('/ai/image_temp', methods=['POST'])
def receive_frame():
    frame_data = request.form.get('frame')
    # Decode the Base64 data
    # For example, if you're using Python's base64 module
    import base64
    decoded_frame = base64.b64decode(frame_data)

    # Process the frame data as needed
    # ...

    return jsonify({'message': "Frame received successfully!"})

if __name__ == '__main__':
    run_class = General.Run()
    app.run(debug=True)
