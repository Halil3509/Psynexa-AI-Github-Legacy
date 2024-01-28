import cv2
import numpy as np
import base64
from .emotionModel import PsynexaModel
import logging
from tqdm import tqdm
import pandas as pd

class EmotionDetection():
    def __init__(self):
        self.plot_dict = None
        self.resutl_dict = dict()
        self.psynexa_model = PsynexaModel()
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
    
    def single_predict(self, img):

        results = self.psynexa_model.detect_emotion_for_single_frame(img)
        
        #self.logger.info("Emotion Detection Processes has been completed.")
        
        return results
    
    
    def api_detection(self, video_array):
        """_summary_
        
        result => 
        plot dict
        plot_dict["time"]
        plot_dict["value"]
        
        """
        try:
            self.logger.info("Emotion API detection process is starting...")
        
            counter = 1
            plot_dict = dict()
            plot_dict["time"] = []
            plot_dict["value"] = []
            for base64_img in tqdm(video_array):
                
                results = self.single_predict(base64_img=base64_img)
                
                print("emotion results: ", results)
                plot_dict["time"].append(counter)
                plot_dict["value"].append(results[0]["emo_label"])
                counter +=1
                
            self.logger.info("Emotion API detection process finished.")
        
            self.plot_dict = pd.DataFrame(plot_dict)

        except Exception as err:
            raise Exception(err)
        


    def detect_video(self, video_path, fps=1):

        if patience == None:
            patience = fps * 2

        counter = 1
        plot_dict = dict()
        plot_dict['time'] = []
        plot_dict['value'] = []

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        general_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the video

        if general_fps == 0:
            general_fps = 1

        video_minutes = int(total_frames / general_fps)

        # We took indexes that equal fps.  But it starts from 1. Because we can select -1 of min value of selected_frame_indexes
        selected_frame_indexes = np.random.randint(1, 30, size=fps)
        total_tqdm_frames = video_minutes * fps


        # We use fps_counter to use for every index of selected_frame_indexes
        fps_counter = 0

        pbar = tqdm(total=total_tqdm_frames, desc='Processing Frames')

        self.logger.info("The Emotion detecting process is starting...")
        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            results = self.single_predict(base64_img=frame)
                
            print("emotion results: ", results)
            plot_dict["time"].append(counter)
            plot_dict["value"].append(results[0]["emo_label"])
                
            self.logger.info("Emotion detection process finished.")
        
            self.plot_dict = plot_dict    

            counter += 1
            fps_counter += 1

            # Update progress bar and indexes
            pbar.update(1)

        pbar.close()
        cap.release()

        return self.plot_dict