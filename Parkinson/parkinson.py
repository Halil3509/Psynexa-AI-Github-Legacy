import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import logging
import cv2

class ParkinsonDetection():
    def __init__(self, model_path = None, type = 'spiral'):
        self.ratio = None,
        self.estimated_label = None,
        self.model = self.get_model(model_path=model_path, type= type)
        self.classes = ['healthy', 'parkinson']
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger

    
    def get_model(self, model_path, type):
        """_summary_

        Args:
            model_path (_type_, optional): _description_. Defaults to None.
            type (str, optional): can be 'spiral' and 'wave'. Defaults to 'spiral'.
        """
        
        if type == 'spiral' and model_path is None:
            model_path ='~/Psynexa-AI-Github/Parkinson/models/spiral_model.hdf5'

        elif type == 'wave' and model_path is None:
            model_path = '~/Psynexa-AI-Github/Parkinson/models/wave_model.hdf5'
            
        else:
            self.logger.error("Type only can be spiral or wave")
        
        self.logger.info(f"{type} model is loading ...")
        model = load_model(model_path)
        self.logger.info(f"{type} model loaded succesfully.")
        return model


    def model_predict(self, img_path, input_shape = (224,224)):
        """_summary_

        Args:
            img_path (_type_): _description_
            input_shape (tuple, optional): _description_. Defaults to (224,224).
        """
        
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized_image = cv2.resize(img, input_shape)
        expanded_image = np.expand_dims(resized_image, axis=0)
        sigmoid_out = self.model.predict(expanded_image)

        if sigmoid_out[0][0] < 0.50:
            label = self.classes[0]
            ratio = np.round(1 - sigmoid_out[0][0], 2)
        else:
            label = self.classes[1]
            ratio = np.round(sigmoid_out[0][0], 2)
            
        self.ratio = ratio
        self.estimated_label = label
        
        self.logger.info("Predicting process was completed :). Let's see results")
        self.logger.info(f"Label = {label} - Ratio(%) = {ratio}")
    
        return label, ratio