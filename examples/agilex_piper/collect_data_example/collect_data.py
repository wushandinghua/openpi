"""Example for collect data functions with agilex piper.

This example involves Piper initialization and starting a state machine
to manage data acquisition, visualization, and storage.

conda create -n piper_run_env python=3.10
conda activate piper_run_env
pip install -i https://pypi.mirrors.ustc.edu.cn/simple airbot_data-1.2.1-py3-none-any.whl
pip install python-can
git clone -b feature/arm_zero_191 https://github.com/agilexrobotics/piper_sdk.git
cd piper_sdk
pip install .
"""

from transitions import Machine
from datetime import datetime, timezone
from pynput import keyboard
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import time
import platform
from PIL import Image
import numpy as np
import copy
import os
import concurrent.futures
import json
import shutil
import cv2
import queue
import threading
import yaml
import math
from piper_sdk import C_PiperInterface_V2 as PiperArm
import pprint
from typing import Dict
from PIL import Image
from typing import Protocol, Dict, List, Optional, Union, List
from dataclasses import dataclass, field, replace
import traceback
import pyudev


# Monkey patch，自动忽略非法参数 like preset
_original_save = Image.Image.save


def safe_save(self, fp, format=None, **params):
    if "preset" in params:
        print("Found 'preset' in save() call — printing call stack:")
        traceback.print_stack()
        del params["preset"]
    return _original_save(self, fp, format=format, **params)


Image.Image.save = safe_save


class DataFileManager:
    """Manages the file operations related to the data collection task.

    The class is responsible for managing data format processing and storage during the
    data collection process. Such as creating directories, managing cache for images and
    low-dimensional data, and organizing the data based on the provided task name and root path.
    It also maintains the state for each task, including the current episode, camera
    settings, and the associated timestamps.

    Attributes:
        task_path (Path): The path to the directory for the current task.
        num_image_writers_per_camera (int): The number of image writers per camera.
        cam_cache (list): A cache for storing camera information.
        current_episode (int): The current episode number.
        low_dim (dict): A dictionary for low-dimensional data.
        images_timestamp (dict): A dictionary for timestamps of images.
        low_dim_timestamps (dict): A dictionary for timestamps of low-dimensional data.
        low_dim_keys (dict): A dictionary for keys of low-dimensional data.
        low_dim_time_keys (dict): A dictionary for time-based keys of low-dimensional data.
        cameras_keys (dict): A dictionary for camera keys.
        dicts_cache (list): A cache for storing dictionaries.
        joint_num (int): The number of joints.
    """

    def __init__(self, root_path: Path, task_name: str) -> None:
        """Initializes a new instance of the DataFileManager class.

        This method sets up the task path by combining the provided root path
        and task name. It also creates the necessary directories if they do not
        already exist. Several internal caches and state variables are initialized.

        Args:
            root_path (Path): The root directory where task files will be stored.
            task_name (str): The name of the task which will be used to create a
                             subdirectory under the root path.

        Returns:
            None: This function does not return any value.
        """
        self.task_path = Path(root_path) / task_name
        if not os.path.exists(self.task_path):
            os.makedirs(self.task_path)
        self.num_image_writers_per_camera = 2

        self.cam_cache = []
        self.current_episode = 0
        self.low_dim = {}
        self.images_timestamp = {}
        self.low_dim_timestamps = {}
        self.low_dim_keys = {}
        self.low_dim_time_keys = {}
        self.cameras_keys = {}
        self.dicts_cache = []
        self.joint_num = 6

    def save_image(img: np.ndarray, frame_index: int, images_dir: Path) -> None:
        """Saves the given image as a PNG file.

        Args:
            img (np.ndarray): The image to be saved, in numpy array format.
            frame_index (int): The index of the current frame, used to generate the file name.
            images_dir (Path): The directory where the image will be saved.

        Returns:
            None: This function does not return any value.

        This function saves the image as a PNG file in the specified directory with a filename
        based on the frame index (e.g., frame_000001.png). The function also ensures that the
        necessary parent directories are created if they do not already exist.
        """
        img = Image.fromarray(img)
        path = images_dir / f"frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)

    def init_keys(self, observation, camera_keys) -> None:
        """Initializes various keys used for accessing data in the observation.

        Args:
            observation (dict): A dictionary containing the observation data, including keys for
                                low-dimensional data, images, and camera settings.
            camera_keys (list): A list of keys representing the cameras.

        Returns:
            None: This function does not return any value.

        This function initializes keys for low-dimensional data, camera settings, and image data
        based on the provided observation. It also determines the number of leader and follower
        robots based on the joint position data, and loads a template dictionary from a JSON file
        depending on the number of leaders. Finally, it checks the current episode and increments
        it to avoid name conflicts when saving data.
        """
        self.low_dim_keys = list(observation["low_dim"].keys())
        self.low_dim_time_keys = [
            key for key in observation["low_dim"] if "time" in key
        ]
        # TODO : get cameras keys
        self.cameras_keys = camera_keys  # robot.cameras
        self.image_keys = [key for key in observation if "image" in key]
        self.cam_num = len(self.image_keys)
        # Get leader/follower num
        self.leader_num = int(
            len(observation["low_dim"]["observation/arm/joint_position"])
            / self.joint_num
        )
        self.follower_num = int(
            len(observation["low_dim"]["action/arm/joint_position"]) / self.joint_num
        )
        print(self.leader_num)
        if self.leader_num == 1:
            with open("./replay1.json") as file:
                self.template_dict = json.load(file)
            print("aaaaaaaaaaaa")
        elif self.leader_num == 2:
            with open("./replay2.json") as file:
                self.template_dict = json.load(file)
        while os.path.exists(self.task_path / f"{self.current_episode}"):
            self.current_episode += 1

    def add_raw_data(self, observation: dict, current_frame: int) -> None:
        """Adds raw data to internal storage structures.

        This method processes and stores raw data from an observation dictionary into internal
        data structures such as low-dimensional data, image timestamps, and image data.

        Args:
            observation (dict): A dictionary containing the raw data from the environment.
                The dictionary contains keys for low-dimensional data, image data, and time-related information.
            current_frame (int): The current frame index. This is used for appending data
                to the appropriate lists.

        Returns:
            None: This function does not return any value.

        The method updates internal data structures:
            - Adds low-dimensional data to `self.low_dim`
            - Adds timestamps for the images to `self.images_timestamp`
            - Appends image data to `self.cam_cache`
        """
        for key in self.low_dim_keys:
            if key not in self.low_dim:
                self.low_dim[key] = []
            self.low_dim[key].append(observation["low_dim"][key])

        for key in self.cameras_keys:
            if key not in self.images_timestamp:
                self.images_timestamp[key] = []
            self.images_timestamp[key].append(observation["/time/" + key])

        for key in self.image_keys:

            img = observation[key]

            # 类型检查：是否为 numpy.ndarray
            if not isinstance(img, np.ndarray):
                print(
                    f"add_raw_data [WARNING] Frame {current_frame} - {key} is not a numpy array. Got type: {type(img)}"
                )
                continue  # 或 raise ValueError(...) 终止执行

            # 可选：形状检查，确保是三通道图像
            if img.ndim != 3 or img.shape[2] not in [3, 4]:
                print(
                    f"add_raw_data [WARNING] Frame {current_frame} - {key} has unexpected shape: {img.shape}"
                )
                continue  # 或 raise

            self.cam_cache.append(observation[key])

    def convert_data(self) -> None:
        """Converts raw data into a structured format for further processing.

        This method takes the raw data stored in internal structures and converts it into a
        structured format that follows a predefined template. The data is organized for
        leader and follower robots, as well as for image data, and timestamps are added.

        Returns:
            None: This function does not return any value.

        This method performs the following:
            - Extracts keys for action, observation, and image data from the template.
            - Converts time-related keys and updates the timestamps.
            - Fills the structured template with low-dimensional data for the arm and end-effector (eef) positions,
            poses, and actions for both leader and follower robots.
            - Adds image data for each camera into the structured template.

        It is assumed that the template is pre-loaded and that there is an existing
        structure for each type of data (e.g., joint positions, end-effector poses).
        """
        arm_keys = [
            k
            for k, v in self.template_dict["data"].items()
            if "action" in k or "observation" in k
        ]
        cam_keys = [k for k, v in self.template_dict["data"].items() if "image" in k]
        frame_num = len(self.low_dim[self.low_dim_keys[0]])

        for key in self.low_dim_time_keys:
            self.low_dim_timestamps[key.replace("/time", "")] = self.low_dim.pop(key)
        if len(self.low_dim_time_keys) == 1:
            self.low_dim_timestamps = self.low_dim_timestamps.popitem()[1]

        for key in arm_keys:
            self.template_dict["data"][key] = [
                copy.deepcopy(self.template_dict["data"][key][0])
                for _ in range(frame_num)
            ]
            for i in range(len(self.template_dict["data"][key])):
                self.template_dict["data"][key][i]["t"] = int(
                    self.low_dim_timestamps[i] * 1000
                )
        for key in cam_keys:
            self.template_dict["data"][key] = [
                copy.deepcopy(self.template_dict["data"][key][0])
                for _ in range(frame_num)
            ]

        for cam_index in range(self.cam_num):
            for i in range(
                len(self.template_dict["data"][f"/images/cam{cam_index+1}"])
            ):
                self.template_dict["data"][f"/images/cam{cam_index+1}"][i]["t"] = int(
                    self.images_timestamp[f"cam{cam_index+1}"][i] * 1000
                )

        for i in range(frame_num):
            for leader_index in range(self.leader_num):
                self.template_dict["data"][
                    f"/observation{leader_index+1}/arm/joint_position"
                ][i]["data"]["pos"] = self.low_dim["observation/arm/joint_position"][i][
                    leader_index * self.joint_num : (leader_index + 1) * self.joint_num
                ]
                self.template_dict["data"][
                    f"/observation{leader_index+1}/eef/joint_position"
                ][i]["data"]["t"] = [
                    self.low_dim["observation/eef/joint_position"][i][leader_index]
                ]
                self.template_dict["data"][f"/observation{leader_index+1}/eef/pose"][i][
                    "data"
                ]["t"] = self.low_dim["observation/eef/pose"][i][
                    leader_index * self.joint_num : 3 + leader_index * self.joint_num
                ]
                self.template_dict["data"][f"/observation{leader_index+1}/eef/pose"][i][
                    "data"
                ]["r"] = self.low_dim["observation/eef/pose"][i][
                    3
                    + leader_index * self.joint_num : 7
                    + leader_index * self.joint_num
                ]
            for follower_index in range(self.follower_num):
                self.template_dict["data"][
                    f"/action{follower_index+1}/arm/joint_position"
                ][i]["data"]["pos"] = self.low_dim["action/arm/joint_position"][i][
                    follower_index
                    * self.joint_num : (follower_index + 1)
                    * self.joint_num
                ]
                self.template_dict["data"][
                    f"/action{follower_index+1}/eef/joint_position"
                ][i]["data"]["t"] = [
                    self.low_dim["action/eef/joint_position"][i][follower_index]
                ]
                self.template_dict["data"][f"/action{follower_index+1}/eef/pose"][i][
                    "data"
                ]["t"] = self.low_dim["action/eef/pose"][i][0:3][
                    follower_index * self.joint_num : 3
                    + follower_index * self.joint_num
                ]
                self.template_dict["data"][f"/action{follower_index+1}/eef/pose"][i][
                    "data"
                ]["r"] = self.low_dim["action/eef/pose"][i][3:][
                    3
                    + follower_index * self.joint_num : 7
                    + follower_index * self.joint_num
                ]
            """
            for cam_index in range(self.cam_num):
                self.template_dict["data"][f"/images/cam{cam_index+1}"][i][
                    "data"
                ] = self.cam_cache[self.cam_num * i + cam_index]
                # self.template_dict['data']['/images/cam2'][i]['data'] = self.cam_cache[2*i+1]
            """
            for cam_index in range(self.cam_num):
                # 计算出当前帧的对应图像在 cam_cache 中的索引
                img = self.cam_cache[self.cam_num * i + cam_index]

                # 写入到模板结构中对应的位置
                self.template_dict["data"][f"/images/cam{cam_index+1}"][i]["data"] = img

                # 打印图像类型确认：确保是 numpy.ndarray
                print(
                    f"[DEBUG_convert_data] 写入后 Frame {i} cam{cam_index+1} data 类型: {type(img)}, 形状: {getattr(img, 'shape', 'N/A')}"
                )

    def save_data(self) -> None:
        """Converts data and saves it to a BSON file.

        This method converts the current raw data using `convert_data` and then saves
        it to a BSON file in the appropriate directory. It also creates a new directory
        for the current episode if it does not already exist.

        The saved BSON file is stored in the format:
            <task_path>/<current_episode>/data.bson

        Returns:
            None: This function does not return any value.

        The `current_episode` is incremented after the data is saved.
        """
        self.convert_data()  # 这一步没问题？
        # Save dict to bson file
        if not os.path.exists(self.task_path / f"{self.current_episode}"):
            os.makedirs(self.task_path / f"{self.current_episode}")
        from airbot_data.io import save_bson

        print("start saving bason")
        # 在这里检查template_dict的图像类型

        # 遍历 template_dict 中的所有图像项，打印类型和形状
        for cam_index in range(self.cam_num):
            cam_key = f"/images/cam{cam_index+1}"
            if cam_key not in self.template_dict["data"]:
                print(f"[WARNING] {cam_key} not found in template_dict['data']")
                continue
            for i, frame in enumerate(self.template_dict["data"][cam_key]):
                img = frame.get("data", None)
                print(
                    f"[Save_Bason_DEBUG] cam{cam_index+1} frame {i} type: {type(img)}, shape: {getattr(img, 'shape', 'N/A')}"
                )
                # 也可以加入类型断言
                assert isinstance(
                    img, np.ndarray
                ), f"[ERROR] cam{cam_index+1} frame {i} is not a numpy array! Got: {type(img)}"

        save_bson(
            self.template_dict, self.task_path / f"{self.current_episode}" / "data.bson"
        )
        print("Save data to ", self.task_path / f"{self.current_episode}" / "data.bson")
        self.current_episode += 1

    def save(self) -> None:
        """Converts data and appends it to the cache.

        This method converts the current raw data using `convert_data` and appends
        the result to the `dicts_cache` list. It does not create a new file or directory.

        Returns:
            None: This function does not return any value.

        The data is stored in the `dicts_cache` for later use or batch saving.
        """
        self.convert_data()
        self.dicts_cache.append(self.template_dict)

    def save_last(self) -> None:
        """Saves all cached data to BSON files.

        This method iterates through the data stored in the `dicts_cache` list and
        saves each entry as a BSON file. A new directory is created for each episode
        if it does not already exist.

        Returns:
            None: This function does not return any value.

        The saved BSON files are stored in the format:
            <task_path>/<episode_index>/data.bson
        """
        from airbot_data.io import save_bson

        for episode_index in range(len(self.dicts_cache)):
            if not os.path.exists(self.task_path / f"{episode_index}"):
                os.makedirs(self.task_path / f"{episode_index}")
            save_bson(
                self.template_dict, self.task_path / f"{episode_index}" / "data.bson"
            )
            print("Save data to ", self.task_path / f"{episode_index}" / "data.bson")

    def clear_cache(self) -> None:
        """Clears all cached data and reloads the template.

        This method clears all the internal caches, including camera images, low-dimensional
        data, and timestamps. After clearing the cache, it reloads the template from a
        default replay file (`replay1.json`).

        Returns:
            None: This function does not return any value.

        The caches are cleared to free up memory or reset data for a new episode.
        """
        self.cam_cache = []
        self.low_dim = {}
        self.images_timestamp = {}
        self.low_dim_timestamps = {}
        self.low_dim_keys = {}
        self.low_dim_time_keys = {}
        self.cameras_keys = {}
        with open("./replay1.json") as file:
            self.template_dict = json.load(file)

    def save_all_images(self, observation: dict, current_frame: int) -> None:
        """Saves all images in the observation using multi-threading.

        This method saves images from the provided `observation` dictionary to disk
        using multi-threading for efficiency. Each image is saved in its corresponding
        directory within the `task_path`.

        Args:
            observation (dict): A dictionary containing image data. Each key corresponds
                to a camera and the associated image data.
            current_frame (int): The current frame index used to name the saved image files.

        Returns:
            None: This function does not return any value.

        Multi-threading is used to save images concurrently for all cameras and frames.
        """
        futures = []
        # Save Images With Muti-Thread
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.image_keys) * self.num_image_writers_per_camera
        ) as executor:
            for key in self.image_keys:
                tmp_imgs_dir = self.task_path / f"{self.current_episode}" / key
                if not os.path.exists(tmp_imgs_dir):
                    os.makedirs(tmp_imgs_dir)
                futures += [
                    executor.submit(
                        self.save_image,
                        observation[key],
                        current_frame,
                        tmp_imgs_dir,
                    )
                ]

    def remove_last_episode(self) -> None:
        """Removes the directory of the last episode.

        This method removes the directory corresponding to the last episode and its
        contents from the file system. It also decrements the `current_episode` index.

        Returns:
            None: This function does not return any value.

        If the directory cannot be removed (e.g., due to an error), an error message
        is printed.
        """
        rm_path = self.task_path / f"{self.current_episode-1}"
        try:
            shutil.rmtree(rm_path)
            print(
                f"Directory {rm_path} and its contents have been removed successfully"
            )
            self.current_episode -= 1
        except OSError as e:
            print(f"Error: {rm_path} : {e.strerror}")


class KeyHandler:
    """Handles key press events for controlling data collection and robot states.

    This class listens for specific key presses and triggers actions in the associated
    `Machine` object. These actions include starting or stopping data collection, printing
    current states, removing the last saved episode, and controlling the robot's behavior.
    """

    def __init__(self, collector: Machine) -> None:
        """Initializes the KeyHandler with the given Machine instance.

        Args:
            collector (Machine): The Machine instance responsible for data collection and state management.
        """
        self.collector = collector

    def show_instruction(self) -> None:
        """Displays the instructions for the key press actions.

        This function provides a user-friendly guide to inform the user about the available
        key press actions for controlling the system.
        """
        print(
            """(Press:
            'Space Bar' to start recording the data,
            'q' to discard current recording or rerecording the last episode,
            'p' to print current arms' states,
            'r' to remove the last saved episode,
            'z' to exit program when robot in Wait state
            'i' to show this instructions again.
        )"""
        )

    def handle_keypress(self, key: keyboard.Key) -> None:
        """Handles key press events and triggers the appropriate actions.

        This function listens for key presses and initiates the corresponding actions in the
        `Machine` instance. The actions can include starting or stopping data collection,
        printing states, removing episodes, or changing robot states.

        Args:
            key: The key event triggered by the user.

        Returns:
            None: This function does not return any value.
        """
        # 根据按键事件触发状态机的状态
        try:
            if key == keyboard.Key.space:
                collect_thread = threading.Thread(target=self.collector.CollectBegin)
                collect_thread.daemon = True
                collect_thread.start()
                self.collector.change_flag = True
            elif key.char == "q":
                ct_thread = threading.Thread(target=self.collector.CollectTermination)
                ct_thread.daemon = True
                ct_thread.start()
                self.change_flag = True
            elif key.char == "p":
                print("Current State: ", self.collector.state)
                print("Current change log: ", self.collector.change_flag)
                print("Current episode: ", self.collector.file_manager.current_episode)
            elif key.char == "r":
                if self.collector.file_manager.current_episode > 0:
                    if self.collector.state == "Wait":
                        self.collector.file_manager.remove_last_episode()
                    else:
                        print(self.collector.state, " do not allow remove!")
                print("Current episode: ", self.collector.file_manager.current_episode)
            elif key.char == "0":
                self.collector.trigger("Resetting")
            elif key.char == "z":
                self.collector.trigger("Exit")
                self.change_flag = True
            elif key.char == "i":
                self.show_instruction()
            elif key.char == "s":
                ct_thread = threading.Thread(target=self.collector.CollectOver)
                ct_thread.daemon = True
                ct_thread.start()
            else:
                print("Invalid key pressed")

        except Exception as e:
            print("ERROR: ", e)


class Visualizer:
    """Handles the visualization of demonstration information and camera data.

    This class manages displaying text information (such as episode and step numbers)
    and images (from camera feeds) in a graphical interface using OpenCV. It updates
    the displayed information and images in real time.
    """

    def __init__(self) -> None:
        """Initializes the Visualizer instance with default settings.

        This constructor sets up the window size, font settings, queue, and initial
        text for episode and steps. It also initializes the image canvas for display.

        Attributes:
            height (int): The height of the display window.
            width (int): The width of the display window.
            font (int): Font type for displaying text.
            font_scale_up (float): Font scale for the upper text (episode).
            font_scale_down (float): Font scale for the lower text (steps).
            thickness_up (int): Thickness of the upper text.
            thickness_down (int): Thickness of the lower text.
            frame_queue (queue.Queue): Queue for storing camera frames to be displayed.
            lock (threading.Lock): Lock for synchronizing access to shared data.
            image (np.ndarray): Image canvas for displaying the background.
            text_top (str): Text to display for the episode information.
            text_bottom (str): Text to display for the step information.
            episode (int): Current episode number.
            steps (int): Current number of steps.
        """
        # Set font properties for text display
        self.height, self.width = 400, 600
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_up = 1
        self.font_scale_down = 3
        self.thickness_up = 1
        self.thickness_down = 2
        self.frame_queue = queue.Queue(maxsize=1)  # Set queue size to 1
        self.lock = threading.Lock()

        # Initialize a white image background
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.text_top = "Episode: 0"
        self.text_bottom = "Steps: 0"

        # Calculate text size for centering the text
        self.text_width_top, self.text_height_top = cv2.getTextSize(
            self.text_top, self.font, self.font_scale_up, self.thickness_up
        )[0]
        self.text_width_bottom, self.text_height_bottom = cv2.getTextSize(
            self.text_bottom, self.font, self.font_scale_down, self.thickness_down
        )[0]
        self.episode = 0
        self.steps = 0

    def show_info_on_image(self) -> None:
        """Displays the current episode and step information on the image.

        This function overlays the episode and step text on the image background,
        and then displays the image with the updated text using OpenCV.
        """
        with self.lock:
            episode = self.episode
            steps = self.steps
        text_top = f"Episode: {episode}"
        text_bottom = f"Steps: {steps}"
        image_copy = self.image.copy()

        # Calculate positions to center the text on the image
        x_top = (self.width - self.text_width_top) // 2
        y_top = int(self.height * 0.25)

        x_bottom = (self.width - self.text_width_bottom) // 2
        y_bottom = int(self.height * 0.75)

        # Add the episode and step text to the image
        cv2.putText(
            image_copy,
            text_top,
            (x_top, y_top),
            self.font,
            self.font_scale_up,
            (0, 0, 255),
            self.thickness_up,
        )
        cv2.putText(
            image_copy,
            text_bottom,
            (x_bottom, y_bottom),
            self.font,
            self.font_scale_down,
            (0, 255, 0),
            self.thickness_down,
        )
        # Show the image with the updated text
        with self.lock:
            cv2.imshow("Demonstration Information", image_copy)
        cv2.waitKey(1)

    def show_cameras(self) -> None:
        """Continuously displays camera data from the frame queue.

        This function runs in a separate thread and keeps displaying the latest
        frames from the camera in real time. It updates the image windows for each
        camera key available in the observation.
        """
        while True:
            self.show_info_on_image()  # Display information on the image
            try:
                # Retrieve the latest frame from the queue (timeout after 1 second)
                observation = self.frame_queue.get(timeout=1)
                image_keys = [key for key in observation if "image" in key]
                for key in image_keys:
                    with self.lock:
                        # Display each camera feed in a separate window
                        cv2.imshow(
                            key, cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                        )
            except queue.Empty:
                pass  # If no new frame, just skip
                # print("No new frame to display")
                # continue
            finally:
                cv2.waitKey(1)  # Allow OpenCV to update windows

    def update_info(self, episode: int, steps: int) -> None:
        """Updates the episode and step information.

        This method is called to update the current episode and step count in the
        visualizer, and it uses a lock to ensure thread-safe access to the shared data.

        Args:
            episode (int): The new episode number.
            steps (int): The new step count.
        """
        with self.lock:
            self.episode = episode
            self.steps = steps


class OpenCVCamera:
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    camera = OpenCVCamera(camera_index=0)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    camera = OpenCVCamera(0, fps=30, width=1280, height=720)
    camera = connect()  # applies the settings, might error out if these settings are not compatible with the camera

    camera = OpenCVCamera(0, fps=90, width=640, height=480)
    camera = connect()

    camera = OpenCVCamera(0, fps=90, width=640, height=480, color_mode="bgr")
    camera = connect()
    ```
    Attributes:
        camera_index (int): Index of the camera device.
        fps (Optional[int]): Frames per second (FPS) to set for the camera.
        width (Optional[int]): Width of the captured frames.
        height (Optional[int]): Height of the captured frames.
        color_mode (str): Color mode of the captured frames ('rgb' or 'bgr').
        camera (cv2.VideoCapture): OpenCV video capture object.
        is_connected (bool): Flag indicating if the camera is connected.
        thread (Optional[threading.Thread]): Thread for asynchronous reading.
        stop_event (Optional[threading.Event]): Event to stop the reading thread.
        color_image (Optional[np.ndarray]): Latest color image captured.
        logs (Dict[str, Any]): Logs for performance and timestamp information.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Initializes the OpenCVCamera object with provided camera configuration.

        Args:
            config (dict): A dictionary containing camera configuration parameters.
        """
        config = SimpleNamespace(**config)
        self.camera_index = config.camera_index
        self.camera_usb_hardware_index = config.camera_usb_hardware_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.video_cameras = {}  # 存储打开后的摄像头对象
        self.context = pyudev.Context()
        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

    # 通过usb硬件配置接口连接USB摄像头,链接成功返回摄像头句柄
    def find_rgb_video_device_by_path(self, usb_path: str):
        for device in self.context.list_devices(subsystem="video4linux"):
            id_path = device.get("ID_PATH")
            dev_node = device.device_node
            if id_path and usb_path in id_path:
                print(f" 尝试打开 {dev_node} (ID_PATH: {id_path})")
                cap = cv2.VideoCapture(dev_node)
                if cap.isOpened():
                    print(f" 成功打开 {dev_node} 对应 USB 接口 {usb_path}")
                    return cap  # 一旦成功就退出！
                else:
                    print(f" 无法打开 {dev_node}(USB: {usb_path})")
        return None

    def connect(self) -> None:
        """
        连接摄像头，并根据配置设置分辨率、帧率、颜色模式等参数。

        如果连接失败，会抛出异常（ValueError 或 OSError），提示用户摄像头是否存在或参数是否设置失败。
        """

        # 如果已经连接了，就不允许重复连接
        if self.is_connected:
            raise ValueError(f"OpenCVCamera({self.camera_index}) 已经连接过了。")
        """
        #第一步：尝试用给定的 camera_index 检查摄像头是否可用
        if platform.system() == "Linux":
            # Linux 平台下，摄像头通常是 /dev/videoX 格式的路径
            tmp_camera = cv2.VideoCapture(f"/dev/video{self.camera_index}")
        else:
            # Windows / Mac 平台直接用索引号
            tmp_camera = cv2.VideoCapture(self.camera_index)
        """

        # 检查摄像头是否打开成功
        # is_camera_open = tmp_camera.isOpened()

        # 马上释放临时对象，避免占用资源
        # del tmp_camera

        # 如果摄像头无法打开，尝试提示更清晰的错误信息
        """
        if not is_camera_open:
            # 查看当前系统中可用的摄像头编号
            available_cam_ids = self.find_camera_indices()
            if self.camera_index not in available_cam_ids:
                # 提示使用者传了错误的编号
                raise ValueError(
                    f"`camera_index` 应该是可用的摄像头编号之一 {available_cam_ids}，但你传的是 {self.camera_index}。\n"
                    "请检查摄像头是否插好，或尝试运行 `python lerobot/common/robot_devices/cameras/opencv.py` 来检测摄像头。"
                )

            # 如果编号是正确的但还是打不开，就抛出连接异常
            raise OSError(f"无法访问 OpenCVCamera({self.camera_index})。")
        """

        # 第二步：正式建立连接（刚才只是试探性验证）
        """
        if platform.system() == "Linux":
            self.camera = cv2.VideoCapture(f"/dev/video{self.camera_index}")
        else:
            self.camera = cv2.VideoCapture(self.camera_index)
        """
        # 使用usb硬件接口序号链接摄像头
        self.camera = self.find_rgb_video_device_by_path(self.camera_usb_hardware_index)
        # 第三步：设置摄像头参数（如果有提供）
        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 读取实际被设置成功的参数值
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 第四步：校验参数是否真的设置成功（OpenCV 有时会“假装”设置成功）
        if self.fps is not None and not math.isclose(
            self.fps, actual_fps, rel_tol=1e-3
        ):
            raise OSError(
                f"设置帧率失败：期望 {self.fps}，实际是 {actual_fps}（摄像头 {self.camera_index}）"
            )
        if self.width is not None and self.width != actual_width:
            raise OSError(
                f"设置宽度失败：期望 {self.width}，实际是 {actual_width}（摄像头 {self.camera_index}）"
            )
        if self.height is not None and self.height != actual_height:
            raise OSError(
                f"设置高度失败：期望 {self.height}，实际是 {actual_height}（摄像头 {self.camera_index}）"
            )

        # 第五步：设置视频编码格式（MJPG 可大幅提升帧率 & 减少 CPU 压力）
        # if self.camera_usb_hardware_index == "usb-0:2.2":
        #     self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        #     print("设置 MJPG 编码")
        # else:
        #     self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

        # 更新成员变量记录实际成功的参数值
        self.fps = actual_fps
        self.width = actual_width
        self.height = actual_height

        # 标记状态为“已连接”
        self.is_connected = True

    def read(self, temporary_color_mode: Optional[str] = None) -> np.ndarray:
        """Read a frame from the camera and return the frame in the format (height, width, channels).

        Args:
            temporary_color_mode (Optional[str]): Temporary color mode for the frame ('rgb' or 'bgr').

        Returns:
            np.ndarray: The captured color image frame.

        Raises:
            ValueError: If the color mode is invalid.
            OSError: If the camera cannot capture an image or if the image size does not match the expected size.
        """
        if not self.is_connected:
            raise ValueError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        ret, color_image = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color_mode = (
            self.color_mode if temporary_color_mode is None else temporary_color_mode
        )

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = datetime.now(timezone.utc)

        return color_image

    def read_loop(self) -> None:
        """Continuously capture frames in a separate thread."""
        while self.stop_event is None or not self.stop_event.is_set():
            self.color_image = self.read()

    def async_read(self) -> np.ndarray:
        """Asynchronously capture a frame in a separate thread and return it when available.

        Returns:
            np.ndarray: The captured color image frame.

        Raises:
            ValueError: If the camera is not connected.
            Exception: If the thread for asynchronous reading fails to start.
        """
        if not self.is_connected:
            raise ValueError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (
                self.thread.ident is None or not self.thread.is_alive()
            ):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        return self.color_image

    def disconnect(self) -> np.ndarray:
        """Disconnect the camera, stop the thread, and release the resources."""
        if not self.is_connected:
            raise ValueError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.release()
        self.camera = None

        self.is_connected = False

    def __del__(self) -> None:
        """Ensure proper cleanup of resources when the object is destroyed."""
        if getattr(self, "is_connected", False):
            self.disconnect()

    def find_camera_indices(
        raise_when_empty=False, max_index_search_range=60
    ) -> List[int]:
        """
        Finds available camera indices by scanning the system for connected cameras.

        On Linux, it scans the '/dev/video*' ports to find the camera indices.
        On macOS and Windows, it tries camera indices from 0 to `max_index_search_range`.

        Args:
            raise_when_empty (bool): Whether to raise an exception if no camera is found. Default is False.
            max_index_search_range (int): The maximum index range to search for cameras on non-Linux platforms (default is 60).

        Returns:
            list[int]: A list of available camera indices.

        Raises:
            OSError: If no cameras are found and `raise_when_empty` is set to True.

        Notes:
            - On Linux, the function looks for camera devices in the '/dev' directory (e.g., '/dev/video0', '/dev/video1', etc.).
            - On non-Linux platforms, it checks camera indices from 0 up to `max_index_search_range`.
            - Cameras that are accessible through OpenCV are identified by attempting to open each camera index.
        """
        if platform.system() == "Linux":
            # Linux uses camera ports
            print(
                "Linux detected. Finding available camera indices through scanning '/dev/video*' ports"
            )
            possible_camera_ids = []
            for port in Path("/dev").glob("video*"):
                camera_idx = int(str(port).replace("/dev/video", ""))
                possible_camera_ids.append(camera_idx)
        else:
            print(
                "Mac or Windows detected. Finding available camera indices through "
                f"scanning all indices from 0 to {60}"
            )
            possible_camera_ids = range(max_index_search_range)

        camera_ids = []
        for camera_idx in possible_camera_ids:
            camera = cv2.VideoCapture(camera_idx)
            is_open = camera.isOpened()
            camera.release()

            if is_open:
                print(f"Camera found at index {camera_idx}")
                camera_ids.append(camera_idx)

        if raise_when_empty and len(camera_ids) == 0:
            raise OSError(
                "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
                "or your camera driver, or make sure your camera is compatible with opencv2."
            )

        return camera_ids

def get_joint_positions(joint_state):
    joints = [joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
              joint_state.joint_4, joint_state.joint_5, joint_state.joint_6]
    return joints

def get_gripper_position(gripper_state):
    return gripper_state.grippers_angle

class PiperPlay:
    """
    A class to manage the Airbot robot and cameras for data collection and robotic control.
    This class handles the initialization, mode switching, and data capture from multiple cameras
    and robotic arms (leader and follower robots).
    """

    def __init__(self, config: Dict[str, any], **kwargs: any) -> None:
        """
        Initializes the PiperPlay object by connecting to cameras and robots.

        Args:
            config: Configuration object containing the necessary parameters (e.g., camera settings, robot IPs).
            **kwargs: Additional arguments passed to the class constructor (not used here).
        """
        self.config = config
        self.cameras = self.config.cameras
        # Initialize cameras
        for name in self.cameras:
            self.cameras[name] = OpenCVCamera(self.cameras[name])
        self.logs = {}
        self.__init()
        # 初始化线程列表
        self.follower_threads = []
        self.follower_stop_flags = []
        self.is_open_follower = []

    def __init(self) -> None:
        """
        初始化 leader 和 follower 机器人，只测试机械臂。
        """
        args = self.config
        leader_robot = []
        follower_robot = []
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        # Initialize leader robots
        for i in range(args.leader_number):
            leader_robot.append(
                PiperArm(can_name=args.leader_can[i])
            )
            time.sleep(0.1)
            print(f"leader robot {i} 初始化成功")

        for i in range(args.follower_number):
            follower_robot.append(
                PiperArm(can_name=args.follower_can[i])
            )
            time.sleep(0.1)
            print(f"follower robot {i} 初始化成功")

        # 初始化时全部设为 False，表示尚未开启跟随
        self.follower_following_flags = [False for _ in range(args.follower_number)]

        self.leader_robot = leader_robot
        self.follower_robot = follower_robot
        time.sleep(0.3)
        # self.reset()

    def reset(self):
        """
        设置控制模式和初始关节位置
        """
        # 先链接上全部的机械臂
        for i, robot in enumerate(self.leader_robot):
            robot.ConnectPort()
            # robot.MasterSlaveConfig(0xFA, 0, 0, 0)
            # robot.MotionCtrl_2(0x01, 0x01, 0, 0, 0, 0x03)
            while( not robot.EnablePiper()):
                time.sleep(0.01)

        for i, robot in enumerate(self.follower_robot):
            robot.ConnectPort()
            # robot.MasterSlaveConfig(0xFC, 0, 0, 0)
            # robot.MotionCtrl_2(0x01, 0x01, 0, 0, 0, 0x02)
            while( not robot.EnablePiper()):
                time.sleep(0.01)
        # 停止跟随（如果有的话）
        self.stop_followers()

        # 移动机械臂到指定位置
        for i, robot in enumerate(self.leader_robot):
            robot.ReqMasterArmMoveToHome(mode=2)
            print(f"leader {i} and follower {i} moved to zero position")
            time.sleep(1)
            robot.ReqMasterArmMoveToHome(mode=0)


    def follower_start(self, leader, follower, delay: float = 0.01):
        """
        启动 follower 跟随 leader 的动作，返回线程并保存退出控制标志。
        """
        # 设置从臂高跟随模式
        follower.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
        follower.EnableArm()

        stop_event = threading.Event()

        def follow_loop():
            while not stop_event.is_set():
                target_joints = leader.GetArmJointCtrl()
                follower.JointCtrl(*target_joints)
                follower.GripperCtrl(leader.GetArmGripperCtrl())
                time.sleep(0.01)  # 控制频率为100hz

        thread = threading.Thread(target=follow_loop, daemon=True)
        thread.start()

        self.follower_threads.append(thread)
        self.follower_stop_flags.append(stop_event)

        print("Follower started following in background.")

    # 跟随模式退出函数
    def stop_followers(self):
        """
        停止所有 follower 的跟随线程，并切换到常规关节控制 模式。
        """
        for i, stop_flag in enumerate(self.follower_stop_flags):
            stop_flag.set()  # 设置标志让线程自动退出
            self.follower_following_flags[i] = False  # 表示第 i 个 follower 开始跟随
            print(f"Stopping follower {i} follow thread...")

        # for i, robot in enumerate(self.follower_robot):
        #     robot.MotionCtrl_2(0x01, 0x01, 50)
        #     robot.EnableArm()
        #     print(f"Follower {i} switched to joint control mode")

        self.follower_threads = []
        self.follower_stop_flags = []

    def enter_active_mode(self) -> None:
        """
        Enters the active mode, where leader robots are controlled actively.
        """
        self.stop_followers()
        self._state_mode = "active"

    def enter_passive_mode(self) -> None:
        """
        Enters the passive mode, where leader robots are controlled manually (preparation for demonstration).
        """
        # args = self.config
        # # 同时启动从机械臂的跟随模式
        # for i in range(args.follower_number):
        #     self.follower_following_flags[i] = True  # 表示第 i 个 follower 开始跟随
        #     self.follower_start(self.leader_robot[i], self.follower_robot[i])
        self._state_mode = "passive"

    def clear_boundary_error(self) -> None:
        """
        Clears any boundary errors and switches to passive mode.
        """
        self.enter_passive_mode()

    # 获取机器人当前状态（低维度数据）
    def get_low_dim_data(self) -> Dict[str, any]:
        """
        收集 leader 和 follower 机器人的低维状态数据。
        """
        args = self.config
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        data = {}
        data["/time"] = time.time()

        # Leader (action)
        action_arm_jq = []
        action_eef_jq = []
        action_eef_pose = []
        # print("开始获取 Leader 机器人状态...")
        for i in range(args.leader_number):
            # print(f"→ Leader {i} joint_position: {leader_robot[i].get_joint_pos()}")
            # print(f"→ Leader {i} end_position: {leader_robot[i].get_eef_pos()}")
            # print(f"→ Leader {i} pose: {leader_robot[i].get_end_pose()}")
            joints_ctrl_msg = leader_robot[i].GetArmJointCtrl()
            action_arm_jq.extend(get_joint_positions(joints_ctrl_msg.joint_ctrl))  # 获取各个关节的位置
            gripper_ctrl_msg = leader_robot[i].GetArmGripperCtrl()
            action_eef_jq.append(
                get_gripper_position(gripper_ctrl_msg.gripper_ctrl)
            )  # 获取夹爪的张开幅度（如果没有夹爪或者夹爪没准备好，返回None）
            pose = [[0] * 3, [0] * 4] #获取末端的空间位姿
            action_eef_pose.extend(pose[0] + pose[1])  # xyz + quat
        data["action/arm/joint_position"] = action_arm_jq
        data["action/eef/joint_position"] = action_eef_jq
        data["action/eef/pose"] = action_eef_pose

        # Follower (observation)
        obs_arm_jq = []
        obs_eef_jq = []
        obs_eef_pose = []
        # print("获取 Follower 机器人状态...")
        for i in range(args.follower_number):
            # print(f"→ Follower {i} joint_position: {follower_robot[i].get_joint_pos()}")
            # print(f"→ Follower {i} end_position: {follower_robot[i].get_eef_pos()}")
            # print(f"→ Follower {i} pose: {follower_robot[i].get_end_pose()}")
            join_msg = follower_robot[i].GetArmJointMsgs()
            obs_arm_jq.extend(get_joint_positions(join_msg.joint_state))  # 获取各个关节的位置
            gripper_msg = follower_robot[i].GetArmGripperMsgs()
            obs_eef_jq.append(get_gripper_position(gripper_msg.gripper_state))  # 获取夹爪的张开幅度
            pose = [[0] * 3, [0] * 4] #获取末端的空间位姿
            obs_eef_pose.extend(pose[0] + pose[1])
        data["observation/arm/joint_position"] = obs_arm_jq
        data["observation/eef/joint_position"] = obs_eef_jq
        data["observation/eef/pose"] = obs_eef_pose

        return data

    # 从相机读取图像帧（图像观察数据） ， 从机器人读取当前状态（低维度状态数据）
    def capture_observation(self) -> Dict[str, any]:
        """
        Captures observations including both images from cameras and low-dimensional data.

        Returns:
            A dictionary containing the time stamps, low-dimensional data, and camera images.
        """
        # 动作列表和图像列表?
        obs_act_dict = {}
        images = {}

        # self.file_manager.cam_cache.clear()

        for name in self.cameras:
            before_camread_t = time.perf_counter()
            img = self.cameras[name].async_read()  # 异步读取图像
            # print(f"[DEBUG] capture_observation: {name} type: {type(img)} shape: {getattr(img, 'shape', None)}") #这边打印的图像类型是没有问题的
            images[name] = img  # 将图像置入列表中

            # 新增：存入 cam_cache
            # self.file_manager.cam_cache.append(img)

            obs_act_dict[f"/time/{name}"] = time.time()  # 添加时间戳
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionaries
        obs_act_dict["low_dim"] = low_dim_data  # 同时记录机械臂低维度的数据
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[
                name
            ]  # 将读取到的图像转存储到obs列表中（要在这里检查图像类型吗）
        return obs_act_dict

    def exit(self) -> None:
        """
        Disconnects the cameras and cleans up resources.
        """
        self.enter_active_mode()
        # 移动机械臂到指定位置
        for i, robot in enumerate(self.leader_robot):
            robot.ReqMasterArmMoveToHome(mode=2)
            print(f"leader {i} and follower {i} moved to zero position")
            time.sleep(1)
            robot.ReqMasterArmMoveToHome(mode=0)
        print("Robot exited")

    def get_state_mode(self) -> str:
        """
        Returns the current state mode of the system (either "active" or "passive").

        Returns:
            The current state mode as a string.
        """
        return self._state_mode


class DataCollecter:
    """
    Class to handle the process of robot data collection, state transitions, and file management.
    It initializes the robot, handles different states like waiting, collecting, saving, and manages the
    collection of data through states and actions.

    Attributes:
        transitions (list[dict]): Defines the state transitions for the finite state machine.
        states (list[dict]): Defines the available states for the state machine.
        config (SimpleNamespace): Configuration object for initialization parameters.
        machine (Machine): State machine for managing the different states.
        file_manager (DataFileManager): Object for handling file operations like saving data.
        visualizer (Visualizer): Object for handling data visualization.
        logger (Any): Logger for logging events, defaults to None.
        max_frames (int): Maximum number of frames to collect.
        num_image_writers_per_camera (int): Number of image writers per camera.
        current_frame (int): Counter for the current frame being processed.
        fps (float): Frames per second for the collection.
        lock (threading.Lock): Lock for thread synchronization.
        change_flag (bool): Flag to indicate when the state should be changed.
    """

    transitions = [
        {"trigger": "Start", "source": "Start", "dest": "Initial"},
        {"trigger": "InitSuccess", "source": "Initial", "dest": "Wait"},
        {"trigger": "CollectBegin", "source": "Wait", "dest": "Collect"},
        {"trigger": "CollectOver", "source": "Collect", "dest": "Save"},
        {"trigger": "CollectTermination", "source": "Collect", "dest": "Wait"},
        {"trigger": "SaveOver", "source": "Save", "dest": "Wait"},
        {"trigger": "Exit", "source": "Wait", "dest": "Exit"},
    ]

    states = [
        {"name": "Start"},
        {"name": "Initial"},
        {"name": "Wait"},
        {"name": "Collect"},
        {"name": "Save"},
        {"name": "Exit"},
    ]

    def __init__(self, config: Dict[str, any], **kwargs) -> None:
        """
        Initialize the DataCollecter object with the provided configuration.

        Args:
            config (dict): Configuration dictionary for initialization.
            kwargs (any): Additional arguments for initialization.
        """
        if config is None:
            raise ValueError("Init config should be not None")
        else:
            self.config = SimpleNamespace(**config)

        # Initialize the state machine
        self.machine = Machine(
            model=self,
            transitions=DataCollecter.transitions,
            states=DataCollecter.states,
            initial="Start",
        )
        self.file_manager = DataFileManager(
            self.config.root_path, self.config.task_name
        )
        self.visualizer = Visualizer()
        self.logger = None
        self.max_frames = self.config.max_frames
        self.num_image_writers_per_camera = self.config.num_image_writers_per_camera
        self.current_frame = 0
        self.fps = self.config.fps
        self.lock = threading.Lock()
        self.change_flag = False
        visualize_thread = threading.Thread(
            target=self.visualizer.show_cameras, daemon=True
        )
        visualize_thread.daemon = True
        visualize_thread.start()

    def RobotInit(self) -> None:
        """
        Initialize the robot and handle any exceptions during initialization.
        """
        try:
            # Initial Robot
            self.robot = PiperPlay(self.config)
            print("Robot Init Success")
        except Exception as e:
            print(f"Robot Init Failed, Because of {e}")
            self.Exit()

    def Wait(self) -> None:
        """
        Handle the 'Wait' state by resetting the robot and clearing the cache.
        Continuously capture observations and push them to the visualizer.
        """
        print("Robot Wait")
        print("Cache Clear")
        # 这里需要将机械臂退出跟随
        self.robot.reset()
        self.current_frame = 0
        self.file_manager.clear_cache()
        while self.state == "Wait":
            observation = self.robot.capture_observation()
            self.visualizer.frame_queue.put(observation)

    def Collect(self) -> None:
        """
        Handle the 'Collect' state by initializing the keys, capturing observations,
        visualizing data, and saving raw data until the maximum frame count is reached.
        """
        print("Robot Collect")
        self.file_manager.init_keys(
            self.robot.capture_observation(), self.robot.cameras
        )
        t1 = time.time()
        self.robot.enter_passive_mode()
        while self.state == "Collect" and self.current_frame < self.max_frames:
            start_loop_t = time.perf_counter()
            print("current: ", self.current_frame)
            with self.lock:
                # Get Observation Data
                observation = (
                    self.robot.capture_observation()
                )  # TODO: Modify method for get data

            # Visualize
            self.visualizer.frame_queue.put(observation)
            self.visualizer.update_info(
                self.file_manager.current_episode, self.current_frame
            )

            # Save Observation Data
            self.file_manager.add_raw_data(
                observation, self.current_frame
            )  # TODO: Camera Data change

            # Update Frame Index
            self.current_frame += 1
            dt_s = time.perf_counter() - start_loop_t
            self.busy_wait(1 / self.fps - dt_s)

        t2 = time.time()
        print("all time: ", t2 - t1)

        # Check to "Save" State
        if self.state == "Collect":
            self.CollectOver()
            self.change_flag = True

    def Save(self) -> None:
        """
        Save dict to bson file
        """
        print("Robot Save")
        self.file_manager.save_data()
        self.SaveOver()
        self.change_flag = True

    def Exit(self) -> None:
        """
        Exit the robot, possibly saving the last data or setting joint positions.
        """
        # TODO : Set joint a fit position
        print("Robot Exit")
        self.robot.exit()

    def busy_wait(self, seconds: Union[int, float]) -> None:
        """
        Busy-wait for the given amount of time, with platform-specific adjustments.

        Args:
            seconds (float): Time to wait in seconds.
        """
        if platform.system() == "Darwin":
            # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
            # but it consumes CPU cycles.
            end_time = time.perf_counter() + seconds
            while time.perf_counter() < end_time:
                pass
        else:
            # On Linux time.sleep is accurate
            if seconds > 0:
                time.sleep(seconds)

    def state_run(self) -> None:
        """
        Run the state machine until the 'Exit' state is reached.
        Continuously checks the current state and executes the corresponding action.
        """
        while self.state != "Exit":
            # print("current state: ", self.state)
            if self.change_flag:
                if self.state == "Wait":
                    self.change_flag = False
                    self.Wait()
                elif self.state == "Collect":
                    self.Collect()
                elif self.state == "Save":
                    self.Save()


if __name__ == "__main__":
    """
    Main entry point for running the data collection process.

    Steps:
    1. Loads configuration from a YAML file.
    2. Initializes the DataCollecter object with the loaded configuration.
    3. Sets up the KeyHandler to handle user inputs and show instructions.
    4. Starts a listener for keyboard events.
    5. Initializes the robot and sets the initial state to 'Wait'.
    6. Runs the state machine until the 'Exit' state is triggered.
    """

    # Load configuration from the YAML file
    with open("config2.yaml") as f:
        # Load the configuration file using YAML
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize the data collector with the loaded configuration
    data_collecter = DataCollecter(config)

    # Initialize the KeyHandler to handle user inputs and show instructions
    key_handler = KeyHandler(data_collecter)
    key_handler.show_instruction()

    # Create a listener for keyboard events, with the on_press handler being the key press handler
    listener = keyboard.Listener(on_press=partial(key_handler.handle_keypress))

    # Start listening for keyboard events in a separate thread
    listener.start()

    # Initialize the robot
    data_collecter.RobotInit()

    # Set the initial state to "Wait" and call the Wait method to handle the initial wait state
    data_collecter.state = "Wait"
    data_collecter.Wait()

    # Run the state machine to process the transitions between states
    try:
        data_collecter.state_run()

    finally:
        # Ensure that the listener is properly stopped and joined, and exit the data collector
        listener.stop()
        listener.stop()
        listener.join()
        data_collecter.Exit()
        print("State machine exited.")
