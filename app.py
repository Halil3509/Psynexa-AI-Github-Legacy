from flask import Flask, request, jsonify
import os 
import base64
import numpy as np
import cv2
import json
from flask_cors import CORS 
import General
import time
# import threading
import requests

app = Flask(__name__)
CORS(app)


# Define a dictionary to store data (in-memory database for this example)
data = {}

run_class = General.Run()

auth_token = """0b6c05e02ee081f0f9d3d733e6dadefcc7d3e5bb2c10f3195927e2794002eefdf5f6f2774afeba9188a133385082a36818baca38f93bf05be5a9c68672a84f3efde436ce64afeedf5e3d79f36980e9e8cd9ed4f41939dd2a666f386118604991d5ada44ca4ca9c02881e1692e8cd5ad4f6016cea4390fb0931ae7c3ae9ad573e"""

headers = {
    'Authorization': f'Bearer {auth_token}',
    'Content-Type': 'application/json'  # Assuming you're sending JSON data
}

image_headers = {
    'Authorization': f'Bearer {auth_token}'
}


# definings
last_request_time = time.time()
frame_buffer = []
eye_scores = []
head_scores = []

emotion_dict = dict()
emotion_dict["time"] = []
emotion_dict["value"] = []

head_plot_dict = dict()
head_plot_dict["time"] = []
head_plot_dict["value"] = []


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
        
        return jsonify({"message": "işlem başarılı",
                        "total_disorder_json": total_disorder_json, 
                        "speech_speed":{
                            "ratio":run_class.speech_speed_class.speech_score,
                            "label": speed_range
                        }
                        })
    else:
        return jsonify({'error': 'No file uploaded'})
    
    

@app.route('/ai/legacy_audio_temp', methods = ['POST'])
def get_audio_temp_legacy():
    print(request.files)
    data = request.json 
    text = data.get("text")
    speed_score = data.get("speech_speed")

    puncted_text = run_class._legacy_punct_model.add_punct(text)
    
    print(puncted_text)
    
    total_disorder_json = run_class.full_disorder_detection_legacy(audio_text=puncted_text)
    
    # Save and clasify speed score
    run_class.speech_speed_class.save_result(speed_score)
    
    
    range_to_disorder = {
            'bipolar': 'Bipolar',
            'depression': 'Depresyon',
            'normal':'Normal'
        }
        
    ## Adding ratio part
    speed_range = run_class.speech_speed_class.speech_range


    disorder_name = range_to_disorder[speed_range]

    # Add speech speed ratio to general
    if disorder_name != 'Normal':
        total_disorder_results_new = General.change_ratios(
            values_dict=total_disorder_json,
            value=run_class.ratios["Speech"][speed_range],
            name=disorder_name,
            type='inc'
        )
    
    return_dict = {
        'data':{
            'generalResult':total_disorder_results_new,
            'speechSpeedLabel': disorder_name
        }
    }
    
    
    # Save DB
    post_url = "http://api.psynexa.com/api/meeting-analyses"
    response_post = requests.post(post_url, headers= headers, json=return_dict)
    created_id = json.loads(response_post.text)['data']['id']
    
    
    put_url = f"http://api.psynexa.com/api/clients/4"
    put_data = {
        "data":{  
                "meetingAnalyzes": [created_id]
            }
    } 
    response_put = requests.put(put_url, headers=headers, json=put_data)
        
    return jsonify({"message": "Legacy Punct text process finished successfully.",
                    "data": response_put.text}), response_put.status_code





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
    
    # Add to DB
    global image_headers
    global headers
    
    files = {'files': ('parkinson_image.png', img_data)}
    image_url = 'http://api.psynexa.com/api/upload'
    response_image = requests.post(image_url, headers= image_headers, files=files)

    data_dict =  json.loads(response_image.text)
    created_id = data_dict[0]["id"]

    created_data = {
        "data": {
            "drawing": created_id,
            "analysis": [{
                "type":"parkinson",
                "label": run_class._spiral_parkinson_class.estimated_label
            }],
            "client": 4
        }
    }
    
    client_result_url =" http://api.psynexa.com/api/client-results"
    response_post = requests.post(client_result_url, headers= headers, json = created_data)
    print("Parkin image was saved in DB")

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
    


def threapy_analyze_and_add():
    
    global frame_buffer
    
    # Run whole analysis
    run_class.analyze_therapy(frame_buffer)
    frame_buffer = []
    
    eye_scores.append(run_class._eye_class.score)
    head_scores.append(run_class._head_class.score["Variance"])
    head_plot_dict["time"].extend(run_class._head_class.plot_values["time"])
    head_plot_dict["value"].extend(run_class._head_class.plot_values["value"])

    
    emotion_dict["time"].extend(run_class._emotion_class.plot_dict["time"])
    emotion_dict["value"].extend(run_class._emotion_class.plot_dict["value"])



def chane_ratio_analyzing_video():
    global frame_buffer
    
    if len(frame_buffer) != 0:
        threapy_analyze_and_add()
    
    
    # Specify Labels
    run_class._eye_class.specify_label(np.mean(eye_scores))
    run_class._head_class.specify_label(np.mean(head_scores))
    
    print("Total disorder results variable is receiving...")
    # Get total results
    get_url = "http://api.psynexa.com/api/meeting-analyses"
    get_response = requests.get(get_url, headers= headers)
    
    data_dict = json.loads(get_response.text)
    
    total_disorder_results_id = data_dict['data'][-1]['id']
    total_disorder_results = data_dict['data'][-1]['generalResult']
    print("Total disorder results received")
    
    
    if len(eye_scores) == 0:
        raise IndexError("There is no image saved.")
    else:
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
        
        head_result_plot_json = [
            {"time": t, "value": v} 
            for t, v in zip(head_plot_dict["time"], head_plot_dict["value"])
        ]
        
        emotion_result_plot_json = [
            {"time": t, "value": v} 
            for t, v in zip(emotion_dict["time"], emotion_dict["value"])
        ]
        
        
        put_url = f"http://api.psynexa.com/api/meeting-analyses/{total_disorder_results_id}"
        
        return_dict =  {'data': {
        "headLabel":run_class._head_class.label,
        "eyeLabel":run_class._eye_class.label,
        "headPlot":head_result_plot_json,
        "emotionPlot":emotion_result_plot_json,
        "generalResult": total_disorder_results
        }}
        
        response_put = requests.put(url = put_url, headers=headers, json=return_dict)
        
        print("Changed ratios for images: ", response_put.text)



@app.route('/ai/video_analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.json  # Assuming the request contains a JSON body
        image = data.get("image")
        
        global frame_buffer
        
        frame_buffer.append(image)
        
        
        if len(frame_buffer) > 20:
            threapy_analyze_and_add()

    
        return jsonify({"message": "Thanks",
                        "Batch size": len(frame_buffer)})
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    

@app.route('/ai/change_video_analyze_ratios', methods=['GET'])
def change_video_ratios():
    try:
        
        chane_ratio_analyzing_video()
        return jsonify({"message": "Total disorders has been updated for images."}), 200
    
    except Exception as err:
        return jsonify({"error": str(err)}), 500
    
    
    
    
@app.route('/ai/send_message_nexa', methods=['POST'])
def chatbot_message():
    data = request.json
    message = data.get("message")
    
    response = run_class._chatbot.predict_api(text=message)
    
    return jsonify({"Nexa": response})

    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
