temp_prompt = """
Zaman zaman kendi duygusal dünyamda kaybolduğumu hissediyorum. Bir an mutlu ve enerjik olabilirim, ancak hemen sonrasında kendimi yalnız, boş ve umutsuz hissediyorum. İlişkilerimde sürekli bir korku ve belirsizlik içinde yaşıyorum, insanların beni terk edeceğinden veya beni anlamayacaklarından endişe ediyorum. Kendi kimliğimle ilgili bir tutarsızlık içindeyim ve sık sık kim olduğumu anlamakta güçlük çekiyorum. Duygusal dalgalanmalarım, günlük yaşamımda ve ilişkilerimde istikrarı sağlamamı zorlaştırıyor. Yardım almak istiyorum, çünkü bu dalgalanmalar beni gerçekten etkiliyor ve normal yaşamımı sürdürmekte zorlanıyorum."

Borderline Kişilik Bozukluğu olan kişiler genellikle duygusal düzenleme, ilişki becerileri ve kendilik algısı üzerinde çalışan terapilerden fayda görebilirler. Tedavi genellikle bireysel terapi, grup terapisi ve bazen ilaç tedavisini içerebilir. Bu ifade, bir hastanın bu bozuklukla başa çıkma süreciyle ilgili duygularını ve zorluklarını yansıtmaktadır.

"""

"""
1. Speech to text
2. Topic Segmentation
3. Disorder Detection 
4. Speech Speed
5. Eye Detection
6. Head Oscillation
7. Emotion
"""
from DisorderDetection import DisorderDetection
from General import calculation
from HeadOscillation.head import Head
from EyeDetection.eye import ADHDCalculation
from SpeechSpeed.speech_speed import SpeechSpeedClass
from TopicSegmentation.gemini_topic import GeminiTopic
from Emotion.emotion import EmotionDetection

gemini_topic_class = GeminiTopic()
speech_speed_class = SpeechSpeedClass()
head_class = Head()
eye_class = ADHDCalculation()

import yaml


def get_ratios(path=r"C:\Users\halilibrahim.hatun\Documents\Psynexa\Capstone-project\General\ratios.yaml"):
    with open(path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return data


ratios = get_ratios()


def analyse_video():
    # Topic Segmentation
    separated_texts = gemini_topic_class.separate_topics(text=temp_prompt)

    # Disorder Detection
    disorder_detection_class = DisorderDetection(topic_texts=separated_texts)
    first_disorder_results = disorder_detection_class.run_disorder_detection()

    # Speech Speed
    speech_speed_class.calculate_speaking_speed(
        audio_path=r"C:\Users\halilibrahim.hatun\Documents\Psynexa\Capstone-project\Recording.wav",
        text=temp_prompt)
    if speech_speed_class.speech_range[0] == 'bipolar':
        new_total_dict = calculation.change_ratios(values_dict=first_disorder_results,
                                                   value=ratios["Speech"][speech_speed_class.speech_range[0]],
                                                   name="Bipolar", type='inc')

    elif speech_speed_class.speech_range == 'depression':
        new_total_dict = calculation.change_ratios(values_dict=first_disorder_results,
                                                   value=ratios["Speech"][speech_speed_class.speech_range[0]],
                                                   name="Depression", type='inc')
    else:
        new_total_dict = first_disorder_results.copy()
        print("Speech speed is normal.")

    print("After Speech: ", new_total_dict)

    # Head Oscillation
    head_class.detect_video(video_path=r"C:\Users\halilibrahim.hatun\Documents\Psynexa\Capstone-project\temp_video.mp4")
    if head_class.label != 'normal':
        new_total_dict = calculation.change_ratios(values_dict=new_total_dict,
                                                   value=ratios["Head"][head_class.label],
                                                   name="DEHB", type="inc")
    print("After Head: ", new_total_dict)

    # Eye ADHD Detection
    eye_class.calc_eye_score(video_path=r"C:\Users\halilibrahim.hatun\Documents\Psynexa\Capstone-project\temp_video.mp4")
    if eye_class.label != 'normal':
        new_total_dict = calculation.change_ratios(values_dict=new_total_dict,
                                                       value=ratios["Eye"][eye_class.label],
                                                       name="DEHB", type="inc")

    print("After Eye: ", new_total_dict)



# Python code to convert video to audio
import moviepy.editor as mp

def convert(mp4_file_path, wav_file_path):
    # Insert Local Video File Path
    clip = mp.VideoFileClip(mp4_file_path)

    # Insert Local Audio File Path
    clip.audio.write_audiofile(wav_file_path)

# Example usage:
mp4_file_path = r"C:\Users\halilibrahim.hatun\Documents\Psynexa\Capstone-project\temp_video.mp4"
wav_file_path = "output_audio.wav"


if __name__ == '__main__':
    # analyse_video()

