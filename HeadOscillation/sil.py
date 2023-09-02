import cv2
import time
from tqdm import tqdm

# Open a video file
cap = cv2.VideoCapture('./HeadVideos/Anomaly.mp4')  # Replace 'video.mp4' with your video file's path

# Calculate the delay time to achieve 2 FPS
frame_rate = 5  # Desired FPS
delay_time = int(1000 / frame_rate)  # Delay time in milliseconds

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the video
print(fps)
# Calculate the number of frames to process based on the desired 2 frames per second
frames_to_process_count = fps
first_index = int(fps/4)
last_index = int((fps*3)/4)

print(f"first_index: {first_index}, last_index: {last_index}")
# Create a tqdm progress bar
with tqdm(total=frames_to_process_count) as pbar:
    #frames_processed = 0

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #print(frame_number)
        
        # time_position = frame_number / fps  # Calculate time position in seconds
        # print(time_position)
        # Process the frame only if it falls within the desired 2 frames per second
        if frame_number == first_index or frame_number == last_index:
            # Process the frame with your model
            #processed_frame = your_model(frame)  # Replace with your model inference code

            # You can use 'processed_frame' for further processing or analysis
            print("girdi")
            # Display the frame (optional)
            cv2.imshow('Video', frame)
            
            first_index += fps
            last_index += fps

            # Update the progress bar
            pbar.update(1)


        print(fps)
        print(f"first_index: {first_index}, last_index: {last_index}")
        # Wait for the specified delay time (in milliseconds)
        #time.sleep(delay_time / 1000)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
