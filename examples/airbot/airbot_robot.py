from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import time
import platform
import numpy as np
import cv2
import threading
import math
from airbot_py.arm import AIRBOTArm as AirbotArm  # 根据你的实际路径修改
from airbot_py.arm import RobotMode, SpeedProfile
from typing import Dict, List, Optional, List
from types import SimpleNamespace
import pyudev
from examples.airbot.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST
#CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST = "cam_high", "cam_left_wrist", "cam_right_wrist"

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
        if self.camera_usb_hardware_index == "usb-0:2.2":
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            print("设置 MJPG 编码")
        else:
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

_ROBOT_CONFIG = {
    "follower_number": 2,
    "follower_ip": ["localhost","localhost"],
    "follower_port": [50053,50054],
    "start_arm_joint_position": [[
        0.38242924213409424, -0.7028687000274658, 1.0111008882522583, 1.4543755054473877, -1.0736628770828247, -1.2277790307998657
    ],
    [
        -0.6158922910690308, -0.6589990258216858, 1.0687037706375122, -1.3025482892990112, 1.350995659828186, 1.342984676361084,
    ]],
    "cameras": {
        CAM_LEFT_WRIST:{
            "camera_index": 2,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:1.1.2"
            },
        CAM_HIGH:{
            "camera_index": 4,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:2.2"
            },
        CAM_RIGHT_WRIST:{
            "camera_index": 6,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:1.3.2"
            }
    }
}

class AIRBOTPlay:
    """
    A class to manage the Airbot robot and cameras for data collection and robotic control.
    This class handles the initialization, mode switching, and data capture from multiple cameras
    and robotic arms (leader and follower robots).
    """

    def __init__(self) -> None:
        """
        Initializes the AIRBOTPlay object by connecting to cameras and robots.

        Args:
            config: Configuration object containing the necessary parameters (e.g., camera settings, robot IPs).
            **kwargs: Additional arguments passed to the class constructor (not used here).
        """
        self.config = SimpleNamespace(**_ROBOT_CONFIG)
        self.cameras = self.config.cameras
        print("airbot config:", self.config)
        # Initialize cameras
        for name in self.cameras:
            self.cameras[name] = OpenCVCamera(self.cameras[name])
        
        self.logs = {}
        self.is_connected = False
        follower_robot = []
        """
        初始化 follower 机器人
        """
        args = self.config  
        for i in range(args.follower_number):
            follower_robot.append(
                AirbotArm(url=args.follower_ip[i], port=args.follower_port[i])
            )
            time.sleep(0.1)
            print(f"follower robot {i} 初始化成功")

        self.follower_robot = follower_robot
        time.sleep(0.3)
        self.is_servo_mode = True
        try:
            self.connect()
        except Exception as e:
            print("Failed to initialize airbot, err:", e)

    def connect(self):
        if self.is_connected:
            print("airbot is already connected. Do not run `robot.connect()` twice.'")
            raise ConnectionError()
        
         # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        """
        设置控制模式和初始关节位置
        """
        args = self.config

        for i, robot in enumerate(self.follower_robot):
            robot.connect()
            # 设置机械臂速度
            robot.switch_mode(RobotMode.PLANNING_POS)
            robot.set_speed_profile(SpeedProfile.SLOW)
            #robot.move_to_joint_pos([0.0] * 6)
            robot.move_to_joint_pos(args.start_arm_joint_position[i])
            robot.move_eef_pos(0.0)
            print(f"follower {i} moved to start position")
        
        for i, robot in enumerate(self.follower_robot):
            # 设置机械臂控制模式
            if self.is_servo_mode:
                robot.switch_mode(RobotMode.SERVO_JOINT_POS)
            robot.set_speed_profile(SpeedProfile.FAST)
        
        self.is_connected = True
    
    def disconnect(self) -> None:
        """
        Disconnects the cameras and cleans up resources.
        """
        if not self.is_connected:
            print("airbot is not connected. You need to run `robot.connect()` before disconnecting.'")
            raise ConnectionError()
        
        try:
            for name in self.cameras:
                self.cameras[name].disconnect()
            args = self.config
            for i in range(self.config.follower_number):
                self.follower_robot[i].switch_mode(RobotMode.PLANNING_POS)
                self.follower_robot[i].set_speed_profile(SpeedProfile.SLOW)
                self.follower_robot[i].move_to_joint_pos([0.0] * 6)
                self.follower_robot[i].move_eef_pos(0.0)
                self.follower_robot[i].disconnect()
        except Exception as e:
            print("Failed to disconnect airbot, err:", e)
        self.is_connected = False
        print("Robot exited")


    # 获取机器人当前状态（低维度数据）
    def get_low_dim_data(self) -> Dict[str, any]:
        """
        收集 leader 和 follower 机器人的低维状态数据。
        """
        args = self.config
        follower_robot = self.follower_robot
        data = {}
        #data["/time"] = time.time()

        # Follower (observation)
        state = []
        # print("获取 Follower 机器人状态...")
        for i in range(args.follower_number):
            # print(f"→ Follower {i} joint_position: {follower_robot[i].get_joint_pos()}")
            # print(f"→ Follower {i} end_position: {follower_robot[i].get_eef_pos()}")
            state.extend(follower_robot[i].get_joint_pos())
            state.extend(follower_robot[i].get_eef_pos())
        data["observation.state"] = state
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

            #obs_act_dict[f"/time/{name}"] = time.time()  # 添加时间戳
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionaries
        obs_act_dict.update(low_dim_data)  # 同时记录机械臂低维度的数据
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[
                name
            ]  # 将读取到的图像转存储到obs列表中（要在这里检查图像类型吗）
        return obs_act_dict
    
    def send_action(self, target_joint_positions):
        assert len(target_joint_positions) // 7 == len(self.follower_robot)
        if not isinstance(target_joint_positions, list):
            target_joint_positions = list(target_joint_positions)
        # 这里假设非阻塞执行，视为并发控制
        for i in range(len(self.follower_robot)):
            if self.is_servo_mode:
                self.follower_robot[i].servo_joint_pos(target_joint_positions[i * 7:i * 7 + 6])
                self.follower_robot[i].servo_eef_pos(target_joint_positions[i * 7 + 6])
                
            else:
                self.follower_robot[i].move_eef_pos(target_joint_positions[i * 7 + 6], blocking=False)
                self.follower_robot[i].move_to_joint_pos(target_joint_positions[i * 7:i * 7 + 6], blocking=False)
        
    def back_home(self):
        args = self.config
        home_positions =  args.start_arm_joint_position[0] + [0.0]
        home_positions = home_positions + args.start_arm_joint_position[1]
        home_positions += [0.0]
        for i in range(self.config.follower_number):
                self.follower_robot[i].set_speed_profile(SpeedProfile.SLOW)
        time.sleep(2)
        self.send_action(home_positions)
        for i in range(self.config.follower_number):
                self.follower_robot[i].set_speed_profile(SpeedProfile.FAST)
        time.sleep(2)
    
    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

if __name__ == "__main__":
    robot = AIRBOTPlay()
    print("obs1:", robot.capture_observation())
    #robot.back_home()
    robot.send_action([0.0] * 6 + [0.2] + [0.0] * 6 + [0.2])
    time.sleep(5)
    robot.send_action([0.0] * 6 + [0.4] + [0.0] * 6 + [0.4])
    time.sleep(5)
    print("obs2:", robot.capture_observation())
    robot.disconnect()