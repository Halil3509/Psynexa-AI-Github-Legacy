import google.generativeai as genai
import logging


class GeminiTopic:
    def __init__(self):
        genai.configure(api_key='AIzaSyBle9ehQtR5jU4NJsq9i6ilmsLdLOoocFo')
        self.logger = self.get_logger()
        self.model = genai.GenerativeModel('gemini-pro')

        self.logger.info("Gemini Model is ready :)")

    @staticmethod
    def get_logger():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        return logging.getLogger(__name__)

    def separate_topics(self, text):
        self.logger.info("Separate topics is starting...")

        prompt = f"prompt = Bu transkripti cümleler aynı kalacak şekilde konularına göre aralarında '-' olacak şekilde ayır. Ekstradan hiçbir şey yazma: f{text}"

        response = self.model.generate_content(prompt)

        separated_texts = response.text.split("-")

        self.logger.info("Separating topics process has been finished.")

        return separated_texts


