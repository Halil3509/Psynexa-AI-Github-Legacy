import numpy as np
import logging
import yaml

import EyeDetection
import HeadOscillation
import SpeechSpeed
import Parkinson
import DemantiaClockTest

class Run():
    def __init__(self, video_path, audio_path, fps = 2):
        self._eye_class = EyeDetection.ADHD_Calculation(video_path=video_path,fps = fps)
        self._head_class = HeadOscillation.Head(video_path=video_path, fps = fps)
        self._speech_speed_class = SpeechSpeed.SpeechSpeedClass(audio_path=audio_path)
        self._spiral_parkinson_class = Parkinson.ParkinsonDetection(type = "spiral")
        self._wave_parkinson_class = Parkinson.ParkinsonDetection(type = "wave")
        self._dementia_clock_test_class = DemantiaClockTest.DemantiaClockTestClass() # model_path parameter have already arranged
        self._ratios = self.get_ratios()
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
    
    def get_ratios(self, path = r"D:\\Psynexa-AI-Github\\General\\ratios.yaml"):
        with open(path, 'r') as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
            
            self.logger.info("ratios has been read")
            return data
      
    
    
    def analyze_therapy(self):
        """
        Emotion, Eye, Head, Speech Speed, Disorder calculations processes
        """
        
        # Eye Score
        self._eye_class.calc_eye_score()

        # Head Score
        self._head_class.detect_video()
        
        # Speech Speed
        self._speech_speed_class.calculate_speaking_speed()
        
        # Emotion that will be added
        
        
        # Disorder Calculation
        
    
    def statistical_touch(self):
        
        # we will be performed after topic segmentation
        
        return 
        
            
    