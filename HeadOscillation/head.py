import base64
import logging
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import plotly.express as px
import plotly.io as pio
import yaml
from tqdm import tqdm

from HeadOscillation import utils


class Head:
    def __init__(self):
        self.score = None
        self.label = None
        self.plot_values = None
        self._full_landmarks = self.get_yaml(name="Landmark")
        self._range = self.get_yaml(
            path=r"/home/psynexa/Psynexa/AI/Psynexa-AI-Github/HeadOscillation/head_range.yaml",
            name="Range YAML")

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger

    def get_video(self, video_path):
        if os.path.exists(video_path):
            return video_path
        else:
            raise FileNotFoundError(f"{video_path} was not found.")

    def get_yaml(self,
                 path=r"/home/psynexa/Psynexa/AI/Psynexa-AI-Github/HeadOscillation/indexes.yaml",
                 name=None):
        with open(path, 'r') as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)

            if name == "Landmark":
                full_landmarks = data['HEAD_INDEXES'] + data["CHIN_INDEXES"] + data["RIGHT_EAR"] + data["LEFT_EAR"]
                self.logger.info(f"{name} is ready for Head :)")
                return full_landmarks

            self.logger.info(f"{name} is ready for Head :)")
            return data

    def save_plot(self, fig, name):
        FOLDER_PATH = './HeadOscillation/ResultPlots'
        if not os.path.exists(FOLDER_PATH):
            os.mkdir(FOLDER_PATH)

        file_path = os.path.join(FOLDER_PATH, f"{name}.png")
        pio.write_image(fig, file_path)

        self.logger.info(f"Your plot has been saved into {FOLDER_PATH}")

    def draw_plot(self, plot_dict, name):
        fig = px.line(x=plot_dict['time'], y=plot_dict['value'], labels={'x': 'Time', 'y': name},
                      title=f'Line Plot for {name}')
        fig.update_yaxes(range=[0, 20])

        self.save_plot(fig, name)

    def detect_video(self, video_path, threshold=3, patience=None, fps=2, name="Normal", plot=False, ):

        if patience == None:
            patience = fps * 2

        counter = 1
        plot_dict = dict()
        plot_dict['time'] = []
        plot_dict['value'] = []

        cap = cv2.VideoCapture(video_path)

        # Load the face detection and face landmark modules from Mediapipe
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        general_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the video

        if general_fps == 0:
            general_fps = 1

        video_minutes = int(total_frames / general_fps)

        # We took indexes that equal fps.  But it starts from 1. Because we can select -1 of min value of selected_frame_indexes
        selected_frame_indexes = np.random.randint(1, 30, size=fps)
        total_tqdm_frames = video_minutes * fps

        ss_situation = False
        ref_counter = 0
        # We use fps_counter to use for every index of selected_frame_indexes
        fps_counter = 0

        pbar = tqdm(total=total_tqdm_frames, desc='Processing Frames')

        self.logger.info("The detecting process is starting...")
        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_number == np.min(selected_frame_indexes) - 1 and ss_situation == False:
                ss_situation = True
                reference_img_landmarks = utils.detect_face_landmarks(frame, mp_face_mesh, mp_face_detection)
                reference_img_modified_array = utils.calc_mean_landmarks(reference_img_landmarks, self._full_landmarks)
                ref_counter = 0
                self.logger.info('Reference Image has been taken firstly.')

            # We are just checking existence of frame number in selected frame that was selected randomly
            if frame_number in selected_frame_indexes and ss_situation:

                landmarks = utils.detect_face_landmarks(frame, mp_face_mesh, mp_face_detection)
                modified_array = utils.calc_mean_landmarks(landmarks, self._full_landmarks)

                deviation = np.abs(reference_img_modified_array - modified_array)

                sum_array = np.sum(deviation)

                if sum_array > threshold:
                    ref_counter += 1
                    if ref_counter >= patience:
                        # Change reference image
                        reference_img_landmarks = utils.detect_face_landmarks(frame, mp_face_mesh, mp_face_detection)
                        reference_img_modified_array = utils.calc_mean_landmarks(reference_img_landmarks,
                                                                                 self._full_landmarks)
                        ref_counter = 0
                        self.logger.info('Reference Image has been changed.')
                else:
                    ref_counter = 0

                plot_dict['value'].append(sum_array)
                plot_dict['time'].append(counter)
                counter += 1
                fps_counter += 1

                # Update progress bar and indexes
                pbar.update(1)

                if fps_counter == len(selected_frame_indexes):
                    selected_frame_indexes += general_fps
                    fps_counter = 0

        pbar.close()
        cap.release()

        self.logger.info("Detecting process finished successfully. Let's save results")

        self.save_results(plot_dict=plot_dict, name=name, plot=plot)
        print("Head score: ", self.score)
        return self.score

    def save_results(self, plot_dict, name=None, plot=False):
        """_summary_

        Args:
            plot_dict (_type_): _description_
            name (_type_, optional): _description_. Defaults to None.
            plot (bool, optional): _description_. Defaults to False.
        
        return => 
        score: {
            "Variance": x,
            "Mean":x,
            "Std":x
        }
        label: str
        """

        result_dict = dict()

        result_dict["Variance"] = np.round(np.var(plot_dict["value"]), 2)
        result_dict["Mean"] = np.round(np.mean(plot_dict["value"]), 2)
        result_dict["Std"] = np.round(np.std(plot_dict["value"]), 2)

        # assign head score
        self.score = result_dict
        self.logger.info("Score is saved into class's head score property.")

        self.specify_label()

        # specify label

        if plot:
            # Save Draw settings
            self.draw_plot(plot_dict=plot_dict, name=name)

    def specify_label(self):

        if self.score["Variance"] < self._range["Head"]["NORMAL"]:
            self.label = "normal"
        elif self.score["Variance"] < self._range["Head"]["MIN_THRESHOLD"]:
            self.label = "low_ADHD"
        elif self.score["Variance"] < self._range["Head"]["NORMAL_THRESHOLD"]:
            self.label = "normal_ADHD"
        elif self.score["Variance"] < self._range["Head"]["HIGH_THRESHOLD"]:
            self.label = "high_ADHD"
        else:
            self.label = "anomaly"

        self.logger.info("Label assigning process has been completed.")
