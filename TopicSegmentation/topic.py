import pandas as pd
import re
import string
import yaml
from gensim import corpora
import logging

class TopicSegmentation():
    def __init__(self, pure_text):
        self.text = pure_text
        self.not_necessary_words = self._get_not_necessary_word()
        self.ftest = open(r"D:\Psynexa-AI-Github\TopicSegmentation\requirements\turkce-stop-words-small.txt",encoding="utf-8").read().split("\n")


    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
        
        
    def _get_not_necessary_word(self, yaml_path = ""):
        with open(r'D:\Psynexa-AI-Github\TopicSegmentation\not_necessary_word_.yaml', 'r') as file:
            # Load the YAML data into a Python variable
            data = yaml.load(file, Loader=yaml.FullLoader)
            
            self.logger.info("Not necessary words has been uploaded")
            return data
        
    
    
    def deEmoji(self, text):

        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        
        self.logger.info("Emoji deleted")
        return str(emoji_pattern.sub('', text) )# no emoji


    def word_tokenize(self, sentence):
        """

        Args:
            sentence (str): any sentence.
        Returns:
            list: each item is a word.
        """


        acronym_each_dot = r"(?:[a-zğçşöüı]\.){2,}"
        acronym_end_dot = r"\b[a-zğçşöüı]{2,3}\."
        suffixes = r"[a-zğçşöüı]{3,}' ?[a-zğçşöüı]{0,3}"
        numbers = r"\d+[.,:\d]+"
        any_word = r"[a-zğçşöüı]+"
        punctuations = r"[a-zğçşöüı]*[.,!?;:]"
        word_regex = "|".join([acronym_each_dot,
                            acronym_end_dot,
                            suffixes,
                            numbers,
                            any_word,
                            punctuations])

        sentence = re.compile("%s"%word_regex, re.I).findall(sentence)
        
        self.logger.info("Word tokenize process completed succesfully.")

        return sentence




    def initial_clean(self, text):
        """
        Function to clean text-remove punctuations, lowercase text etc.
        """
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower() # lower case text
        text = self.word_tokenize(text)

        self.logger.info("Initial clean finished")
        return text


    
    def remove_stop_words(self, text):
        
        result = [word.lower() for word in text if word.lower() not in self.ftest] 
        self.logger.info("Remove stop words process finished")  
        return result


    def apply_all(self, text):
        """
        This function applies all the functions above into one
        """
        
        result = self.remove_stop_words(self.initial_clean(self.deEmoji(text)))
        self.logger.info("Apply all method was finished. First cleaning process done")
        return result
    
    
    # Elimination 1 
    def manual_concatenation(self, text):
        """ If there is at least one same word in two consecutive sentences, these two sentences are combined.

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """

        edited_list = []
        index = 0
        concat_counter = 0


        for i, sent in enumerate(text.split(".")):
            if i == 0:
                edited_list.insert(0, sent)

            elif len(sent) > 0:
                last_tokenized_sent = self.apply_all(edited_list[-1])
                tokenized_sent = self.apply_all(sent)

                if len(set(last_tokenized_sent) & set(tokenized_sent)) > 0:
                    new_sentence = f"{edited_list[-1]}. {sent}"
                    edited_list[index] = new_sentence
                    concat_counter +=1

                else:
                    index += 1
                    edited_list.insert(index, sent)


        self.logger.info("Manual Concatenation progress finished")
        self.logger.info(f"Totally, {concat_counter} times has been performed concatenation")

        return edited_list
    
    
    # Elimination 2 
    def delete_small_sent(self, edited_list, min_length = 20):
        """Delete small size sentences. Small size will be specified by the User

        Args:
            edited_list (_type_): _description_
            min_length (int, optional): _description_. Defaults to 20.

        Returns:
            _type_: _description_
        """
        concatenated_df = pd.DataFrame()
        concatenated_df["Text"] = edited_list

        concatenated_df["Text"] = concatenated_df[concatenated_df["Text"].astype(str).str.len() > min_length]
        concatenated_df.dropna(inplace = True)
        concatenated_df = concatenated_df.reset_index(drop = True)

        self.logger.info("Elmination 2 finished")

        return concatenated_df
    
    
    # Elimination 3 
    def remove_not_necessary_words(self, data, data_text_column_name = "Text", threshold = 30):
        new_data = data.copy()

        removed_counter = 0

        for i, sent in enumerate(data[data_text_column_name].values):
            # remove punct
            no_punct_sent_list = re.sub(r'[^\w\s]', '', sent.lower()).split()

            common_words = set(no_punct_sent_list) & set(self.not_necessary_words)

            ratio = (len(common_words) / len(no_punct_sent_list))*100

            if ratio > threshold:
                new_data.drop(i, axis = 0, inplace = True)
                removed_counter += 1

        self.logger.info("Removed not necessary words progress finished")
        self.logger.info(f"Totally, {removed_counter} rows has been removed")

        return new_data
    

    # Main method
    def convert_text_and_eliminate(self):

        # Elimination 1
        edited_list = self.manual_concatenation(self.text)

        # Elimination 2
        filtered_df = self.delete_small_sent(edited_list)

        # Elimination 3
        more_meaningful_data =  self.remove_not_necessary_words(filtered_df)

        print("Whole converting and elimination processes have been finised successfully.")

        return more_meaningful_data