import cv2
import numpy as np
import base64
from .emotionModel import PsynexaModel
import logging
from tqdm import tqdm

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
    
    
    def single_predict(self, base64_img):
        
        img_data = base64.b64decode(base64_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
        
            self.plot_dict = plot_dict

        except Exception as err:
            raise Exception(err)