import numpy as np
import yaml
import logging
import mediapipe as mp
import cv2
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import statistics
import base64

from EyeDetection import utils


class ADHD_Calculation():
    def __init__(self):
        self.score = None
        self.label = None # Result Label
        self._coords = self.get_yaml_file()
        self._range = self.get_yaml_file(r"D:\\Psynexa-AI-Github\\EyeDetection\\eye_range.yaml")
        
        
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(__name__)
            # Configure logging settings for this logger
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        return self._logger
    
    def get_yaml_file(self, file_path = r'D:\\Psynexa-AI-Github\\EyeDetection\\landmarks.yaml'):
        try:
            with open(file_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
                self.logger.info("Loaded YAML data from file: %s", file_path)
                return yaml_data
        except Exception as e:
            self.logger.error("Error loading YAML file: %s", str(e))
            return None
    
    
    
    def calc_eye_score(self, video_path, plot = False, plot_name = "Normal Plot", fps = 2):
        
        
        camera = cv2.VideoCapture(video_path)

        plot_dict = dict()
        plot_dict['values'] = []
        plot_dict['time'] = []
        frame_counter = 0
        
        total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        general_fps = int(camera.get(cv2.CAP_PROP_FPS))  # Frames per second of the video
        
        if general_fps == 0:
            general_fps = 1
            
        video_minutes = int(total_frames/ general_fps)
        
        selected_frame_indexes = np.random.randint(1,30, size = fps)
        total_tqdm_frames = video_minutes * fps
        fps_counter = 0

        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            with tqdm(total=total_tqdm_frames) as pbar:    
                
                while True:
                    frame_counter +=1 # frame counter
                    ret, frame = camera.read() # getting frame from camera 
                    if not ret: 
                        break # no more frames break
                    
                    frame_number = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    if frame_number in selected_frame_indexes:
                        
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        results  = face_mesh.process(bgr_frame)
                        if results.multi_face_landmarks:
                            mesh_points = utils.landmarksDetection(frame, results)
                
                            # Get Eye mean 
                            left_eye_coords = [mesh_points[p] for p in self._coords['LEFT_EYE']]
                            left_mean_eye_coords = np.mean(left_eye_coords, axis=0)
                
                            right_eye_coords = [mesh_points[p] for p in self._coords['RIGHT_EYE']]
                            right_mean_eye_coords = np.mean(right_eye_coords, axis=0)

                
                            # Iris Mean
                            left_iris_coords = [mesh_points[p] for p in self._coords['LEFT_IRIS']]
                            left_mean_iris_coords = np.mean(left_iris_coords, axis=0)
                
                            right_iris_coords = [mesh_points[p] for p in self._coords['RIGHT_IRIS']]
                            right_mean_iris_coords = np.mean(right_iris_coords, axis=0)
                            
                        
                            # Calculating average of distance
                            distance_right = np.linalg.norm(left_mean_eye_coords - left_mean_iris_coords)
                            distance_left = np.linalg.norm(right_mean_eye_coords - right_mean_iris_coords)
                            
                            # We get focal lengths to put distance in a scale 
                            focal_length_left = self.calc_focal_length(frame, iris_side='left')
                            focal_length_right = self.calc_focal_length(frame, iris_side='right')
                            
                            if focal_length_left is not None and focal_length_right is not None:
                             
                                result_left = np.round((distance_left/focal_length_left)*100, 2)
                                result_right = np.round((distance_right/focal_length_right)*100, 2)
                                
                                result = (result_left + result_right)/2
                                
                                plot_dict['values'].append(result)
                                plot_dict['time'].append(frame_counter)
                            
                            
                        # Increase progress bar
                        pbar.update(1)
                        fps_counter +=1
                        if fps_counter == len(selected_frame_indexes):
                            selected_frame_indexes += general_fps
                            fps_counter = 0
                        
                    
        self.logger.info("Eye calculation process finished :)")
        
        
        if plot:
            self.draw_adhd_plot(plot_dict=plot_dict, name = plot_name)
        
        self.score  = np.round(statistics.variance(plot_dict['values']), 2)
        self.logger.info(f"{plot_name} video score is {self.score} :)")
        
        # Assign Label
        self.specify_label()
         
        return plot_dict    
    
    
    
    
    
    def api_detection(self, video_array, plot = False, plot_name = "Normal Plot"):
        
        self.logger.info("Eye API detection process is starting...")
        
        plot_dict = dict()
        plot_dict['values'] = []
        plot_dict['time'] = []
        frame_counter = 0
        
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
     
     
            for base64_frame in tqdm(video_array):
                
                # Base64 to img
                img_data = base64.b64decode(base64_frame)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results  = face_mesh.process(bgr_frame)
                if results.multi_face_landmarks:
                    mesh_points = utils.landmarksDetection(frame, results)
        
                    # Get Eye mean 
                    left_eye_coords = [mesh_points[p] for p in self._coords['LEFT_EYE']]
                    left_mean_eye_coords = np.mean(left_eye_coords, axis=0)
        
                    right_eye_coords = [mesh_points[p] for p in self._coords['RIGHT_EYE']]
                    right_mean_eye_coords = np.mean(right_eye_coords, axis=0)

        
                    # Iris Mean
                    left_iris_coords = [mesh_points[p] for p in self._coords['LEFT_IRIS']]
                    left_mean_iris_coords = np.mean(left_iris_coords, axis=0)
        
                    right_iris_coords = [mesh_points[p] for p in self._coords['RIGHT_IRIS']]
                    right_mean_iris_coords = np.mean(right_iris_coords, axis=0)
                    
                
                    # Calculating average of distance
                    distance_right = np.linalg.norm(left_mean_eye_coords - left_mean_iris_coords)
                    distance_left = np.linalg.norm(right_mean_eye_coords - right_mean_iris_coords)
                    
                    # We get focal lengths to put distance in a scale 
                    focal_length_left = self.calc_focal_length(frame, iris_side='left')
                    focal_length_right = self.calc_focal_length(frame, iris_side='right')
                    
                    if focal_length_left is not None and focal_length_right is not None:
                        
                        result_left = np.round((distance_left/focal_length_left)*100, 2)
                        result_right = np.round((distance_right/focal_length_right)*100, 2)
                        
                        result = (result_left + result_right)/2
                        
                        plot_dict['values'].append(result)
                        plot_dict['time'].append(frame_counter)
                    
                    
                    
        self.logger.info("Eye calculation process finished :)")
        
        
        if plot:
            self.draw_adhd_plot(plot_dict=plot_dict, name = plot_name)
        
        self.score  = np.round(statistics.variance(plot_dict['values']), 2)
        self.logger.info(f"{plot_name} video score is {self.score} :)")
        
        # Assign Label
        self.specify_label()
         
        return plot_dict    
    
    
    def specify_label(self):
        """
        result => 
        score: number(float)
        label: str
        """
        
        if self.score < self._range["DEHB"]["MIN_THRESHOLD"]:
            self.label = "normal"
        elif self.score >= self._range["DEHB"]["MIN_THRESHOLD"]:
            self.label = "low_ADHD"
        elif self.score < self._range["DEHB"]["NORMAL_THRESHOLD"]:
            self.label = "normal_ADHD"
        else:
            self.label = "high_ADHD"
        
        self.logger.info("Label assigning processes has been completed.")
    
    
    
    def draw_adhd_plot(self, plot_dict, name):
        """
        Draw distribution plot and save it

        Args:
            plot_dict (_type_): adhd values according to time
            name (_type_): plot's title and name that will be saved
        """
        
        fig = plt.figure()
        
        plt.title(f"{name} Distribution Plot")
        plt.plot(plot_dict['time'], plot_dict['values'])
        # plt.ylim(0,2)
        
        self.add_plot_result(fig, name)
        
        
        
    def add_plot_result(self, fig, name):
        """
        Save plot into DistributionResults folder

        Args:
            fig (_type_): plt variable that has been already created.
            name (_type_): plot's title and name that will be saved. 
        """
        folder_path = "./EyeDetection/DistributionResults"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            self.logger.info("DistributionResults folder has been created to keep distribution plots.")
        
        try:
            file_path = os.path.join(folder_path, f"{name}_plot.jpg")
            fig.savefig(file_path)
            self.logger.info(f"{name} has been added into DistributionResults Folder")
            
        except FileExistsError:
            self.logger.error(f"{name} has not been added due to the fact that it had already been existed.")
            
    
    
    
    def live_detection(self, live_cam  = False):
        """
        According to the user's request open live cam or video that user assigned and display in real time. 

        Args:
            live_cam (bool, optional): ıf live_cam is true, The camera will be opened. Otherwise the video will be appeared that the user gave. 
        """
        video_capture_path = self.video_path
        
        if live_cam:
            video_capture_path = 0
        
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        camera = cv2.VideoCapture(video_capture_path)
        frame_counter = 0
        
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

            # starting time here 
            start_time = time.time()
            # starting Video loop here.
            while True:
                frame_counter +=1 # frame counter
                ret, frame = camera.read() # getting frame from camera 
                if not ret: 
                    break # no more frames break

                

                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results  = face_mesh.process(bgr_frame)
                if results.multi_face_landmarks:
                    mesh_points = utils.landmarksDetection(frame, results)

                    # Get eye mean 
                    left_eye_coords = [mesh_points[p] for p in self._coords['LEFT_EYE']]
                    left_mean_eye_coords = np.mean(left_eye_coords, axis=0)

                    right_eye_coords = [mesh_points[p] for p in self._coords['RIGHT_EYE']]
                    right_mean_eye_coords = np.mean(right_eye_coords, axis=0)
                    
                    # Eye segmentation
                    [cv2.circle(frame,mesh_points[p], 1, utils.GREEN , -1, cv2.LINE_AA) for p in self._coords['LEFT_EYE']]
                    [cv2.circle(frame,mesh_points[p], 1, utils.GREEN ,- 1, cv2.LINE_AA) for p in self._coords['RIGHT_EYE']]
                    
                    # Eye mean detection
                    cv2.circle(frame, tuple(left_mean_eye_coords.astype(int)), 1, utils.BLUE , -1, cv2.LINE_AA)
                    cv2.circle(frame, tuple(right_mean_eye_coords.astype(int)), 1, utils.BLUE , -1, cv2.LINE_AA)

                    # Iris Mean
                    left_iris_coords = [mesh_points[p] for p in self._coords['LEFT_IRIS']]
                    left_mean_iris_coords = np.mean(left_iris_coords, axis=0)

                    right_iris_coords = [mesh_points[p] for p in self._coords['RIGHT_IRIS']]
                    right_mean_iris_coords = np.mean(right_iris_coords, axis=0)
                    
                    cv2.circle(frame, tuple(left_mean_iris_coords.astype(int)), 1, utils.RED , -1, cv2.LINE_AA)
                    cv2.circle(frame, tuple(right_mean_iris_coords.astype(int)), 1, utils.RED , -1, cv2.LINE_AA)
                    
                    # Iris segmentation
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self._coords['LEFT_IRIS']])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self._coords['RIGHT_IRIS']])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv2.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
                    cv2.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)


                    # CAlculate difference
                    distance_right = np.linalg.norm(left_mean_eye_coords - left_mean_iris_coords)
                    distance_left = np.linalg.norm(right_mean_eye_coords - right_mean_iris_coords)
                    distance = np.round((distance_left + distance_right)/2, 2)
                    cv2.putText(frame, f"Distance: {distance:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

                # calculating  frame per seconds FPS
                end_time = time.time()-start_time
                fps = frame_counter/end_time

                frame = utils.textWithBackground(frame,f'FPS: {round(fps,1)}',utils.FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)
                # writing image for thumbnail drawing shape
                # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if key==ord('q') or key ==ord('Q'):
                    break
                
        cv2.destroyAllWindows()
        camera.release()
                
    
    def calc_focal_length(self, image, iris_side = "left"):
        """
        

        Args:
            image (_type_): _description_
            iris_side (str, optional): Side of iris. Defaults to "left". 

        Returns:
            _type_: _description_
        """
        width = image.shape[0]
        height = image.shape[1]

        iris_left_min_x = -1
        iris_left_max_x = -1
        
        if iris_side == 'left':
            iris_coords = self._coords['LEFT_IRIS']
        elif iris_side == 'right':
            iris_coords = self._coords['RIGHT_IRIS']
        else:
            raise ValueError("iris_side parameter can only be right or left. Please choose one of them!")
        
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

        
            results = face_mesh.process(image)
        
            if results.multi_face_landmarks:
                mesh_points = utils.landmarksDetection(image, results)
                for point in self._coords['LEFT_IRIS']:
                    point0 = mesh_points[point]
                    if iris_left_min_x == -1 or point0[0] < iris_left_min_x:
                        iris_left_min_x = point0[0]
                    if iris_left_max_x == -1 or point0[0] > iris_left_max_x:
                        iris_left_max_x = point0[0]

            real_dx = iris_left_max_x - iris_left_min_x
            constant_dx = 11.7

            normalizedFocaleX = 1.40625
            fx = np.min([width, height]) * normalizedFocaleX
            if real_dx == 0:
                dZ = None
            else:
                dZ = (fx * (constant_dx / real_dx)) / 10.0
                dZ = np.round(dZ, 2)

            
            return dZ