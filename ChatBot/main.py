from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from googletrans import Translator, LANGUAGES
import logging

class ChatBot():
    def __init__(self, model_name = "facebook/blenderbot-400M-distill"):
        self.model = self.get_model(model_name)
        self.tokenizer = self.get_tokenizer(model_name)
        
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
        translator = Translator()
        translation = translator.translate(text, src=source_lang, dest=target_lang)
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