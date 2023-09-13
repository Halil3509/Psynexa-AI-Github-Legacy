from flask import Flask, request, jsonify
import os 
import base64
import numpy as np
import cv2
import json
from flask_cors import CORS 
import General


app = Flask(__name__)
cors = CORS(app)


# Define a dictionary to store data (in-memory database for this example)
data = {}

run_class = General.Run()


def upload_audio(file):
    # Create a directory to store the uploaded files if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Get the file name
    filename = file.filename
    
    # Save the file to the 'uploads' directory
    print("Şuanki path: ", os.getcwd())
    base_url = 'uploads/'
    file.save(base_url + filename)
    
    # Return the path to the uploaded file
    return base_url + filename



@app.route('/ai/audio_temp', methods = ['POST'])
def get_audio_temp():
    print(request.files)
    uploaded_file = request.files['dosya.m4a']

    # Check if a file was uploaded
    if uploaded_file.filename != '':
        uploaded_path = upload_audio(uploaded_file)

        
        
        print("uploaded path:", uploaded_path)
        total_disorder_json = run_class.full_disorder_detection(audio_path=uploaded_path)
        
        ## Speech Speed
        run_class.speech_speed_class.calculate_speaking_speed(audio_path = uploaded_path, text = run_class.converted_pure_text)
        
        
        range_to_disorder = {
            'bipolar': 'Bipolar',
            'depression': 'Depresyon'
        }
        
        ## Adding ratio part
        speed_range = range_to_disorder[run_class.speech_speed_class.speech_range]
        
        if speed_range in range_to_disorder:
            disorder_name = range_to_disorder[speed_range]

            # Add speech speed ratio to general
            General.change_ratios(
                values_dict=total_disorder_json,
                value=run_class.ratios["Speech"][range_to_disorder[speed_range]],
                name=disorder_name,
                type='inc'
            )
                
        ## Add speech speed ratio to general
        General.change_ratios(values_dict=total_disorder_json,
                              value=run_class.ratios["Speech"][run_class.speech_speed_class.speech_range],
                              name="")
        
        return jsonify({"message": "işlem başarılı",
                        "total_disorder_json": total_disorder_json, 
                        "speech_speed":{
                            "ratio":run_class.speech_speed_class.speech_score,
                            "label": speed_range
                        }
                        })
    else:
        return jsonify({'error': 'No file uploaded'})
    





@app.route('/ai/predict_spiral_parkinson', methods = ['POST'])
def detect_spiral_parkinson():
    data = request.json  
    base64_image = data.get('image')
    disorder_results = data.get('result')
    
    
    # Decode the base64 image
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    run_class._spiral_parkinson_class.model_predict(img)
    
    print("Spiral Parkinson Adding ratio process is starting...")
    print("Disorder results:", disorder_results)
    # Add Ratio
    change_type = 'inc' if run_class._spiral_parkinson_class.estimated_label == "parkinson" else 'dec'
    print("esimated label:", run_class._spiral_parkinson_class.estimated_label)
    updated_disorder_results = General.change_ratios(values_dict=disorder_results, value=run_class.ratios["Parkinson"]["Spiral"][run_class._spiral_parkinson_class.estimated_label],
                                                     name="Parkinson", type=change_type)


    print("Spiral Parkinson Adding ratio process finished")
    print("New Disorder results: ", updated_disorder_results)

    return jsonify({'message': 'Parkinson Spiral process finished successfully',
                    "label": run_class._spiral_parkinson_class.estimated_label,
                    "ratio": str(run_class._spiral_parkinson_class.ratio)})





@app.route('/ai/predict_wave_parkinson', methods = ['POST'])
def detect_wave_parkinson():
    data = request.json  
    base64_image = data.get('image')
    disorder_results = data.get('result')
    
    # Decode the base64 image
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    run_class._wave_parkinson_class.model_predict(img)
    
    print("Wave Parkinson Adding ratio process is starting...")
    print("Disorder results:", disorder_results)
    # Add Ratio
    change_type = 'inc' if run_class._wave_parkinson_class.estimated_label == "parkinson" else 'dec'
    print("esimated label:", run_class._wave_parkinson_class.estimated_label)
    updated_disorder_results = General.change_ratios(values_dict=disorder_results, value=run_class.ratios["Parkinson"]["Wave"][run_class._wave_parkinson_class.estimated_label],
                                                     name="Parkinson", type=change_type)


    print("Wave Parkinson Adding ratio process finished")
    print("New Disorder results: ", updated_disorder_results)
    
    return jsonify({'message': 'Parkinson Wave process finished successfully',
                    "label": run_class._wave_parkinson_class.estimated_label,
                    "ratio": run_class._wave_parkinson_class.ratio})
    



@app.route('/ai/predict_clock_test', methods = ['POST'])
def detect_dementia():
    data = request.json  
    base64_image = data.get('image')
    disorder_results = data.get('result')
    
    # Decode the base64 image
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    run_class._dementia_clock_test_class.model_predict(img)
    
    
    print("Dementia Clock Test Adding ratio process is starting...")
    print("Disorder results:", disorder_results)
    # Add Ratio
    change_type = 'inc' if run_class._dementia_clock_test_class.estimated_label == "dementia" else 'dec'
    print("esimated label:", run_class._dementia_clock_test_class.estimated_label)
    updated_disorder_results = General.change_ratios(values_dict=disorder_results, value=run_class.ratios["Dementia"][run_class._dementia_clock_test_class.estimated_label],
                                                     name="Demans", type=change_type)


    print("Dementia Adding ratio process finished")
    
    
    return jsonify({'message': 'Demetia clock process process finished successfully',
                    "label": run_class._dementia_clock_test_class.estimated_label,
                    "ratio": run_class._dementia_clock_test_class.ratio})    
    


@app.route('/ai/image_temp', methods=['POST'])
def receive_frame():
    try:
        print(request.json)
        json_file = request.json['images']
        
        
            
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
    
    
@app.route('/ai/video_analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.json  # Assuming the request contains a JSON body
        images = data.get("images", [])

        
        # Run whole analyze
        run_class.analyze_therapy(images)
        
        total_disorder_results =  {
            "Agorafobi": 0.1,
            "Bipolar": 0.1,
            "Borderline": 0.1,
            "Cinsel İlişkili Bozukluklar": 0.0,
            "DEHB": 90.9,
            "Demans": 8.1,
            "Depresyon": 0.0,
            "Madde ile ilişkili bozukluklar": 0.1,
            "OKB": 0.3,
            "Paranoid": 0.0,
            "Parkinson": 0.1,
            "Sosyal Fobi": 0.1,
            "Yeme Bozuklukları": 0.0  } # get with request
        
                
        # Adding ratio processes
        
        # Eye
        if run_class._eye_class.label != 'normal':
            total_disorder_results = General.change_ratios(values_dict= total_disorder_results, 
                                  value= run_class.ratios["Eye"][run_class._eye_class.label],
                                name = "DEHB", type = "inc")
    
        # Head
        if run_class._head_class.label != 'normal':
            total_disorder_results = General.change_ratios(values_dict=total_disorder_results,
                                  value=run_class.ratios["Head"][run_class._head_class.label],
                                  name = "DEHB", type = "inc")
         
        
        
        return jsonify({'result': {
            "Message":"Analyzing therapy process finished succesfully",
            "Eye":{
                "eye_label":run_class._eye_class.label,
                "score":run_class._eye_class.score
            },
            "Head":{
                "label":run_class._head_class.label,
                "score":run_class._head_class.score,
                "plot_values":run_class._head_class.plot_values
            },
            "Emotion":{
                "plot_values": run_class._emotion_class.plot_dict
            },
            "General_results": total_disorder_results
        }}), 200
        
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    

    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
