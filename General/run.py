import numpy as np
import logging
import yaml
import torch
import whisper

# import EyeDetection
# import HeadOscillation
#import SpeechSpeed
import Parkinson
import DemantiaClockTest
import TopicSegmentation
import DisorderDetection

class Run():
    def __init__(self, fps = 2):
        # self._eye_class = EyeDetection.ADHD_Calculation(video_path=video_path,fps = fps)
        # self._head_class = HeadOscillation.Head(video_path=video_path, fps = fps)
        self._spiral_parkinson_class = Parkinson.ParkinsonDetection(type = "spiral")
        self._wave_parkinson_class = Parkinson.ParkinsonDetection(type = "wave")
        self._dementia_clock_test_class = DemantiaClockTest.DemantiaClockTestClass() # model_path parameter have already arranged
        self.whisper_model = self.get_whisper_model()
        #self._speech_speed_class = SpeechSpeed.SpeechSpeedClass(whisper_model= self.whisper_model)
        self._ratios = self.get_ratios()
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
    
    def get_whisper_model(self):

        torch.cuda.is_available()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"The device is {DEVICE}. ")

        self.logger.info("Model is loading...")
        model = whisper.load_model("medium", device = DEVICE)
        self.logger.info("Model is ready :)")

        return model
    
    def speech_to_text(self):
        
        # Speech to text process
        self._converted_pure_text = self._whisper_model.transcribe(self._audio_path)['text']
        self.logger.info("Speech to text process has been finisehd succesfully. ")
        
    
    
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
    
    
    def full_disorder_detection(self, audio_path):
        
        self._audio_path = audio_path
        
        
        # Speech to Text Part
        self.speech_to_text()
        
        # Convert text and eliminate  (Topic Side)
        topic_class = TopicSegmentation.TopicSegmentation(self._converted_pure_text)
        topic_texts = topic_class.convert_text_and_eliminate()
        
        # Disorder Part
        disorder_class = DisorderDetection.DisorderDetection(topic_texts=topic_texts)
        total_disorder_result_json = disorder_class.run_disorder_detection()
        
        self.logger.info("Full Disorder Detection process was finished.")
        
        return total_disorder_result_json
   

    
    def statistical_touch(self):
        
        # we will be performed after topic segmentation
        
        return 
        
            
    