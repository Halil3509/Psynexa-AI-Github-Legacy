import cv2
import numpy as np
import base64
from FinalModel import PsynexaModel
import logging

class EmotionDetection():
    def __init__(self):
        self.resutl_dict = dict()
        self.psynexa_model = PsynexaModel()
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
    
    def api_recognition(self, image_url):
        
        image_data = base64.b64decode(image_url.split(",")[1])

        image_content = np.frombuffer(image_data, np.uint8)

        img = cv2.imdecode(image_content, cv2.IMREAD_COLOR)

        results = self.psynexa_model.detect_emotion_for_single_frame(img)
        
        self.logger.info("Emoteion Detection Processes has been completed.")
        
        return results
