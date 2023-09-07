import numpy as np
from keras.models import load_model
import logging
import cv2

class DemantiaClockTestClass():
    def __init__(self, model_path = None):
        self.ratio = None,
        self.estimated_label = None,
        self.model = self.get_model(model_path=model_path)
        self.classes = ['dementia', 'healthy']
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger

    
    def get_model(self, model_path = None):
        """_summary_

        Args:
            model_path (_type_, optional): _description_. Defaults to None.
        """
        model_path = "D:\Psynexa-AI-Github\DemantiaClockTest\models\demantia.hdf5"
        
        self.logger.info("Demantia Clock Test model is loading ...")
        model = load_model(model_path)
        self.logger.info("Demantia Clock Test model loaded succesfully.")
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