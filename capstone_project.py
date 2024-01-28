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

import subprocess
import whisper

gemini_topic_class = GeminiTopic()
speech_speed_class = SpeechSpeedClass()
head_class = Head()
eye_class = ADHDCalculation()
whisper_model = whisper.load_model("medium")
emotion_model = EmotionDetection()


import yaml


def get_ratios(path=r"/home/psynexa/Psynexa/AI/Psynexa-AI-Github/General/ratios.yaml"):
    with open(path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return data


ratios = get_ratios()


def speech_to_text(audio_path):
    print("Speech to text is starting...")
    result = whisper_model.transcribe(audio_path)
    print("Speech to text process has been finished.")
    return result["text"]


def analyse_video(video_path):

    output_audio_path= "/home/psynexa/Psynexa/AI/Psynexa-AI-Github/Results/wav_files/temp_audio.wav"
    # Mp4 to wav
    # command = f"ffmpeg -i {video_path} {output_audio_path}"
    print("Vİdeo is being converted...")
    # result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    convert(video_path, output_audio_path)
    print("Video has been converted to wav file.")
    # Speech to text
    transcript = speech_to_text(output_audio_path)
    print("Transkript:", transcript)

    # Topic Segmentation
    separated_texts = gemini_topic_class.separate_topics(text=transcript)

    # Disorder Detection
    disorder_detection_class = DisorderDetection(topic_texts=separated_texts)
    first_disorder_results = disorder_detection_class.run_disorder_detection()

    # Speech Speed
    speech_speed_class.calculate_speaking_speed(
        audio_path=output_audio_path,
        text=temp_prompt)

    if speech_speed_class.speech_range[0] == 'bipolar':
        new_total_dict = calculation.change_ratios(values_dict=first_disorder_results,
                                                   value=ratios["Speech"][speech_speed_class.speech_range[0]],
                                                   name="Bipolar", type='inc')

    elif speech_speed_class.speech_range[0] == 'depression':
        new_total_dict = calculation.change_ratios(values_dict=first_disorder_results,
                                                   value=ratios["Speech"][speech_speed_class.speech_range[0]],
                                                   name="Depression", type='inc')
    else:
        new_total_dict = first_disorder_results.copy()
        print("Speech speed is normal.")

    print("After Speech: ", new_total_dict)

    # Head Oscillation
    head_class.detect_video(video_path=r"/home/psynexa/Psynexa/AI/Psynexa-AI-Github/temp_video.mp4")
    if head_class.label != 'normal' and head_class.label != 'anomaly':
        new_total_dict = calculation.change_ratios(values_dict=new_total_dict,
                                                   value=ratios["Head"][head_class.label],
                                                   name="DEHB", type="inc")
    print("After Head: ", new_total_dict)

    # Eye ADHD Detection
    eye_class.calc_eye_score(video_path=r"/home/psynexa/Psynexa/AI/Psynexa-AI-Github/temp_video.mp4")
    if eye_class.label != 'normal':
        new_total_dict = calculation.change_ratios(values_dict=new_total_dict,
                                                       value=ratios["Eye"][eye_class.label],
                                                       name="DEHB", type="inc")

    print("After Eye: ", new_total_dict)


    # Emotion Detection
    emotion_plot_values = emotion_model.detect_video(video_path=video_path)




# Python code to convert video to audio
import moviepy.editor as mp

def convert(mp4_file_path, wav_file_path):
    # Insert Local Video File Path
    clip = mp.VideoFileClip(mp4_file_path)

    # Insert Local Audio File Path
    clip.audio.write_audiofile(wav_file_path)


import plotly.express as px
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go

def save_plots(disorder_dict: dict, emotions: pd.DataFrame, head_oscillations: dict):

    # *** General disorder plot ***

    fig = plt.figure(figsize = (15,30))
    # Verileri sırala
    siralama = sorted(disorder_dict.items(), key=lambda x: x[1], reverse=False)
    isimler, degerler = zip(*siralama)

    # Bar plot oluştur
    fig = px.bar(x=degerler, y=isimler, labels={'x':'Oranlar', 'y':'Hastalıklar'}, title='PSikolojik Hastalık Durumu Grafiği')
    # Figür boyutunu artır
    fig.update_layout(width=800, height=600)

    save_file_path_disorder ='/home/psynexa/Psynexa/AI/Psynexa-AI-Github/Results/plots/general_disorder_plot.png'

    # save as an image
    fig.write_image(save_file_path_disorder)
    print("General disorder plot was saved. ")


    # *** Head Oscillation plot ***
    # Rastgele veri oluştur
    random.seed(0)  # Sabit bir seed kullanarak her çalıştırmada aynı veri üretilmesini sağla
    veri = [random.uniform(0, 0.50) for _ in range(1000)]

    # Eğer önceki bir figür varsa temizle
    go.Figure()

    # Line plot oluştur
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=veri, mode='lines', name='Rastgele Veri'))

    # Horizontal çizgi ekle
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            x1=len(veri),
            y0=0.60,
            y1=0.60,
            line=dict(color="red", width=2)
        )
    )

    fig.update_yaxes(range=[0, 2])

    fig.update_layout(
        yaxis_title="Kafa Hareket Değişimi",
        xaxis_title="Zaman"
    )

    save_file_path_head ='/home/psynexa/Psynexa/AI/Psynexa-AI-Github/Results/plots/head_oscillation_plot.png'

    # save as an image
    fig.write_image(save_file_path_head)


    # *** Emotions ***


    # Her bir duygu için belirlenmiş renkler
    duygu_renkleri = {
        "Üzgün": "blue",
        "Kızgın": "red",
        "Mutlu": "yellow",
        "Korkmuş": "purple",
        "Tiksinmiş": "green",
        "Şaşırmış": "orange",
        "Normal": "gray"
    }


    # Scatter plot oluşturma
    scatter_fig = px.scatter(df, x='time', y='emotion', color='emotion',
                            color_discrete_map=duygu_renkleri,
                            title='Zamana Bağlı Duygu Değişimi')

    # Çizgi grafiği oluşturma
    line_fig = px.line(df, x='Zaman', y='Duygu', title='Zamana Bağlı Duygu Değişimi')

    # Her iki grafiği tek bir alt başlık altında birleştirme
    scatter_fig.update_traces(mode='markers', marker=dict(size=10))  # Nokta boyutunu ayarla
    line_fig.update_traces(line=dict(dash='dot'))  # Çizgiyi kesik yap

    for trace in line_fig.data:
        scatter_fig.add_trace(trace)

    # Grafiği göster
    scatter_fig.show()

    return save_file_path_disorder, save_file_path_head, 




if __name__ == '__main__':
    analyse_video(video_path='/home/psynexa/Psynexa/AI/Psynexa-AI-Github/temp_video.mp4')
