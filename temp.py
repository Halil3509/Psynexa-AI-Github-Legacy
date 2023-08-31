import EyeDetection

ADHD_Class = EyeDetection.ADHD_Calculation(video_path= './EyeDetection/VideoExamples/Ahmet.mp4')

ADHD_Class.calc_eye_score(plot = True, plot_name="Ahmet")

