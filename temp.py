# import EyeDetection

# ADHD_Class = EyeDetection.ADHD_Calculation(video_path= './EyeDetection/VideoExamples/Normal.mp4')

# ADHD_Class.calc_eye_score(plot = True, plot_name="Halil1")




import HeadOscillation

Head_class = HeadOscillation.Head(video_path="./HeadOscillation/HeadVideos/Busra.mp4")

Head_class.detect_video(name = "Busra")

print(Head_class.head_score)




# import SpeechSpeed 

# SpeechSpeedClass = SpeechSpeed.SpeechSpeedClass('./SpeechSpeed/SpeechRecording/bipolar1.wav')

# SpeechSpeedClass.calculate_speaking_speed()





# import Parkinson

# Parkinson_detecting_class = Parkinson.ParkinsonDetection(type = "wave")

# Parkinson_detecting_class.model_predict(img_path = "./Parkinson/temp_images/healthy_wave.png")


# import ChatBot

# BlenderBot = ChatBot.ChatBot()

# BlenderBot.live_chat()



# import DemantiaClockTest

# DemantiaClockTestClass = DemantiaClockTest.DemantiaClockTestClass()

# DemantiaClockTestClass.model_predict(img_path='./DemantiaClockTest/temp_images/demantia.png')