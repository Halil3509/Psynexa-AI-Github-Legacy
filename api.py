from flask import Flask, request, jsonify
import os 

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



@app.route('/audio_temp', methods = ['POST'])
def get_audio():
    
    uploaded_file = request.files['dosya.mp4a']

    # Check if a file was uploaded
    if uploaded_file.filename != '':
        uploaded_path = upload_audio(uploaded_file)


        return jsonify({"message": "işlem başarılı", "uploaded_path": uploaded_file})
    else:
        return jsonify({'error': 'No file uploaded'})
    
    

@app.route('/ai/full_disorder_detection', methods = ['POST'])
def get_audio():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    # If the user submits an empty part without a file, the browser
    # will send an empty part without a filename.
    if file.filename == '':
        return 'No selected file'
    
    if file and file.filename.endswith('.m4a'):
        print("M4a file has been received successfully. ")
        print(file.filename)
        return 'File uploaded successfully'
    
    return 'Invalid file format. Please upload an M4A file.'
    

if __name__ == '__main__':
    app.run(debug=True)
