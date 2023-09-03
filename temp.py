import EyeDetection

# ADHD_Class = EyeDetection.ADHD_Calculation(video_path= './EyeDetection/VideoExamples/Ahmet.mp4')

# ADHD_Class.calc_eye_score(plot = True, plot_name="Ahmet")


import HeadOscillation

# Head_class = HeadOscillation.Head(video_path="./HeadOscillation/HeadVideos/DEHB.mp4")

# Head_class.detect_video(name = "DEHB")

# print(Head_class.head_score)

import SpeechSpeed 

SpeechSpeedClass = SpeechSpeed.SpeechSpeedClass('./SpeechSpeed/SpeechRecording/bipolar1.wav')

SpeechSpeedClass.calculate_speaking_speed()
