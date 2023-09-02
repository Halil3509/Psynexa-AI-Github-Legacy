import cv2
import numpy as np



def detect_face_landmarks(frame, mp_face_mesh, mp_face_detection):

    # Initialize the face detection and face landmark models
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh()

    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    results_detection = face_detection.process(frame_rgb)
    landmarks = None
    if results_detection.detections:
        for detection in results_detection.detections:
            ih, iw, _ = frame.shape

            # Get landmarks for the detected face
            landmarks = face_mesh.process(frame_rgb).multi_face_landmarks

    return landmarks


def calc_mean_landmarks(landmarks, selected_indexes):
    """
    """
    modified_array = np.zeros((len(selected_indexes), 2))
    if landmarks:
        landmark = landmarks[0]
        for i, selected_index in enumerate(selected_indexes):
            point = landmark.landmark[selected_index]
            modified_array[i, 0] = point.x
            modified_array[i, 1] = point.y

    return modified_array


