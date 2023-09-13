import pandas as pd
import logging
import torch
import whisper
import speech_recognition as sr
import yaml



class SpeechSpeedClass():
    def __init__(self):
        self.speech_range = None
        self.speech_score = None
        self.range = self.get_range()
        
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger

    def get_range(self, path = r'D:\\Psynexa-AI-Github\\SpeechSpeed\\speed_range.yaml'):
        with open(path, 'r') as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
            
            self.logger.info("Range was taken for speech speed :)")
            return data


    def calculate_speaking_speed(self, audio_path, text):
        
        self.audio_path = audio_path
        
        recognizer = sr.Recognizer()

        with sr.AudioFile(self.audio_path) as source:
            audio = recognizer.record(source)


        try:
            word_count = len(text.split())
            duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)

            speaking_speed = word_count / (duration / 60)  # Words per minute
            
            # Return
            self.save_result(speaking_speed)
            self.logger.info("Speed Speech Score saved in class succesfully. ")
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            return None
    
    
    def save_result(self, result):
        if result < self.range["NORMAL_MIN"]:
            self.speech_range = "depression"
            self.speech_score = result
            
        
        elif result < self.range['HIGH_MIN']:
            self.speech_range = "normal"
            self.speech_score = result
            
        
        else:
            self.speech_range = "bipolar",
            self.speech_score = result
            
        
        
        
