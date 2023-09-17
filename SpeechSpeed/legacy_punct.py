from transformers import pipeline
import logging
import requests

class LegacyPunctuation():
    def __init__(self):
        self.restored_text = None
        self.api_key = "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"
        self.api_url = "https://api-inference.huggingface.co/models/uygarkurt/bert-restore-punctuation-turkish"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
            
    def get_model(self):
        self.logger.info("Punctuation model is loading...")
        pipe = pipeline(task="token-classification", model="uygarkurt/bert-restore-punctuation-turkish")
        self.logger.info("Punctuation model loaded.")
        
        return pipe
    

    @staticmethod
    def replace_char_at_index(input_string, index, new_char):
        if index < 0 or index >= len(input_string):
            return input_string  # Index out of bounds, return the original string unchanged
        else:
            return input_string[:index] + new_char + input_string[index+1:]


    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()



    def add_punct(self, text):

        self.logger.info("Punctuation process is starting...")
        
        punct_results = self.query({
            "inputs": text,
        })
        
        punct_results.sort(key=lambda x: x['end'], reverse=False)
        
        # Add last index value as space
        text= text + ' '
        
        for punct in punct_results:
            print(punct)
            if punct['entity_group'] == "PERIOD":
                end_index = punct['end']
                    
                text = self.replace_char_at_index(text, end_index, ".")
        self.logger.info("Punctuation process finished.")
        return text
        
