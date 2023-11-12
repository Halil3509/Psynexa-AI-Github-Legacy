from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from googletrans import Translator, LANGUAGES
import logging
import requests

class ChatBot():
    def __init__(self, model_name = "facebook/blenderbot-400M-distill", api = True):
        if api:
            self.api_url = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
            self.headers = {"Authorization": "Bearer <api_key>"}
            self.past_user_inputs = ["Hello"]
            self.generated_responses = ["Hello, I am Nexa. How can I help you?"]
        else:
            self.model = self.get_model(model_name)
            self.tokenizer = self.get_tokenizer(model_name)
        
        self.translator = Translator()
            
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
        
    
    def get_model(self, model_name):
        self.logger.info("Chatbot is loading...")
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.logger.info("Chatbot loaded") 
        return model
        
    
    def get_tokenizer(self, model_name):
        self.logger.info("Tokenizer is loading...")
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.logger.info("Tokenizer loaded")
        return tokenizer
    
    
    def translate(self, text, source_lang, target_lang):
        translation = self.translator.translate(text, src=source_lang, dest=target_lang)
        return translation.text
    
    
    def predict(self, text):
        self.logger.info("Chatbot is typing...")
        eng_text = self.translate(text, 'tr', 'en')
        
        inputs = self.tokenizer([eng_text], return_tensors="pt")
        reply_ids = self.model.generate(**inputs)
        result_text = self.tokenizer.batch_decode(reply_ids)
        tr_text = self.translate(result_text[0], 'en', 'tr')
        return tr_text
    
    
    def live_chat(self):
        while True:
            user_input = input("You: ")

            if user_input.lower() == "exit":
                print("Chatbot: Goodbye!")
                break

            # Encode the user input and generate a response
            response_text = self.predict(user_input)

            print(f"Chatbot: {response_text}")
            
    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()
            
            
            
    def predict_api(self, text):
        
        
        user_text_en = self.translator.translate(text, src='tr', dest='en').text

        
        output_en = self.query({
            "inputs": {
                "past_user_inputs": self.past_user_inputs,
                "generated_responses": self.generated_responses,
                "text": user_text_en
            }
        })
        print(output_en)
        if len(output_en['conversation']['past_user_inputs']) > 3:
            self.past_user_inputs = output_en['conversation']['past_user_inputs'][:-3]
            self.generated_responses = output_en['conversation']['generated_responses'][:-3]

        output_tr = self.translator.translate(output_en["generated_text"], src='en', dest='tr').text

        return output_tr
            
        
        
        