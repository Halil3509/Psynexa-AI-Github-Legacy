import torch
import json 
import logging
from transformers import BertTokenizer
import numpy as np

class DisorderDetection():
    def __init__(self, topic_texts):
        self.topic_texts = topic_texts
        self.label_classes = self._get_label_classes()
        self.model = self._get_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = self._get_tokenizer()
        
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
    @staticmethod
    def divide_dict_values(dict, divide_count):
        return {key: np.round(value / divide_count, 2) for key, value in dict.items()}
    
    @staticmethod
    def sum_dicts(dict1, dict2):
        return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}
    
    @staticmethod
    def split_text(text, threshold = 512):
        if len(text) <= threshold:
            return [text]
        
        chunks = []
        current_chunk = ""
        for sentence in text.split("."):
            current_chunk += sentence + "."
            if len(current_chunk) >= threshold:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    
    def chunks_disorder_detection(self, chunks):
        total_result_dict = dict()
        for label in self.label_classes.values():
            total_result_dict[label] = 0
            
        for chunk in chunks:
            new_result_dict = self.predict(chunk, self.model, max_len = len(chunk))

            #sum
            total_result_dict =  self.sum_dicts(total_result_dict, new_result_dict)

        return self.divide_dict_values(total_result_dict, len(chunk))
    
    
    def _get_tokenizer(model_name = "dbmdz/bert-base-turkish-128k-uncased"):
        return BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    
    def _get_model(self, model_path = r"D:\\Psynexa-AI-Github\\DisorderDetection\\models\\bert_alpha_v3.pt"):
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            self.logger.info("DL model is ready :)")
            return model
        except TypeError:
            raise TypeError("DL model must be entire saved model.")
            
            
    def _get_label_classes(self, seslf, path = ""):
        with open('data.json', 'r') as file:
            # Load the JSON data
            data = json.load(file)
            self.logger.info("Label classes was read")
            return data

            
        
    def predict(self, sample_text, max_len = 100):
        """
        Prection process
        """
        encoded_dict = self.tokenizer.encode_plus(
                            sample_text,
                            add_special_tokens = True,
                            max_length = max_len,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(encoded_dict['input_ids'], token_type_ids=None,
                            attention_mask=encoded_dict['attention_mask'])

        logits = outputs[0]
        logits_cpu = logits.cpu()
        logits = logits_cpu.detach().numpy()

        probabilities = np.round(np.exp(logits) / np.exp(logits).sum(), 3)

        # label = np.argmax(probabilities)
        result_dict = dict()
        for index, probability in enumerate(np.squeeze(probabilities)):
            result_dict[self.label_classes[index]] = np.round(probability*100, 2)

        return result_dict
    
    
            
    def run_disorder_detection(self, topic_texts):
        total_result_dict = dict()
        for label in self.label_classes.values():
            total_result_dict[label] = 0
        
        for text in topic_texts:
            if len(text) > 512:
                chunks = self.split_text(text)
                new_result_dict = self.chunks_disorder_detection(chunks)

            else:
                new_result_dict = self.predict(text, self.model, max_len = len(text))

            print(text)
            print("sonuç:", new_result_dict)
            print("*/"*60)
            total_result_dict = self.sum_dicts(total_result_dict, new_result_dict)

        return self.divide_dict_values(total_result_dict, len(topic_texts))
