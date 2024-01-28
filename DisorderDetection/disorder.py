import logging

import numpy as np
import torch
import yaml
import string
from transformers import BertTokenizer


class DisorderDetection:
    def __init__(self, topic_texts):
        self.topic_texts = topic_texts
        self.label_classes = self._get_label_classes()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._get_model()
        self.tokenizer = self._get_tokenizer()

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger

    @staticmethod
    def get_mean_dict_values(dict):
        total = sum(dict.values())
        return {key: np.round((value / total) * 100, 2) for key, value in dict.items()}

    @staticmethod
    def sum_dicts(dict1, dict2):
        return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}

    @staticmethod
    def split_text(text, threshold=512):
        if len(text) <= threshold:
            return [text]

        chunks = []
        current_chunk = ""
        for sentence in text.split("."):
            if len(current_chunk) + len(sentence) < threshold:
                current_chunk += sentence + "."

            else:
                chunks.append(current_chunk.strip())
                current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    @staticmethod
    def is_only_punctuation(input_str):
        return all(char in string.punctuation for char in input_str)

    def chunks_disorder_detection(self, chunks):
        total_result_dict = dict()
        for label in self.label_classes:
            total_result_dict[label] = 0

        for chunk in chunks:
            if len(chunk) > 0 and not self.is_only_punctuation(chunk):
                new_result_dict = self.predict(chunk, max_len=len(chunk))

                # sum
                total_result_dict = self.sum_dicts(total_result_dict, new_result_dict)

        return self.get_mean_dict_values(total_result_dict)

    def _get_tokenizer(self, model_name="dbmdz/bert-base-turkish-128k-uncased"):
        self._logger.info("Tokenizer is uploading ...")
        tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased", do_lower_case=True)
        self.logger.info("Tokenizer has been uploaded.")
        return tokenizer

    def _get_model(self, model_path=r"/home/psynexa/AI/Psynexa-AI-Github/DisorderDetection/models/bert_alpha_v4.pt"):
        self.logger.info("Disorder Detection model is uploading...")
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            self.logger.info("DL model is ready :)")

            self.logger.info("Disorder Detection model has been uploaded.")
            return model
        except TypeError:
            raise TypeError("DL model must be entire saved model.")

    def _get_label_classes(self, path=r"C:\Users\halilibrahim.hatun\Documents\Psynexa\Capstone-project\DisorderDetection\label_classes.yaml"):
        with open(path, 'r') as file:
            # Load the YAML data into a Python variable
            data = yaml.load(file, Loader=yaml.FullLoader)

            self.logger.info("Label classes has been uploaded")
            return data['label_classes']

    def predict(self, sample_text, max_len=100):
        """
        Prection process
        """
        encoded_dict = self.tokenizer.encode_plus(
            sample_text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
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
            result_dict[self.label_classes[index]] = np.round(probability * 100, 2)

        return result_dict

    def run_disorder_detection(self):
        total_result_dict = dict()
        for label in self.label_classes:
            total_result_dict[label] = 0

        for text in self.topic_texts:
            if len(text) > 0:
                if len(text) > 512:
                    chunks = self.split_text(text)
                    new_result_dict = self.chunks_disorder_detection(chunks)

                else:
                    new_result_dict = self.predict(text, max_len=len(text))

                total_result_dict = self.sum_dicts(total_result_dict, new_result_dict)

        return self.get_mean_dict_values(total_result_dict)
