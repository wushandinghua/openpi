import time
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
import numpy as np
import cv2
import rclpy

class Robot(Node):
    def __init__(self):
        super().__init__('robot_node')
        
        # 创建回调组
        self.image_callback_group = ReentrantCallbackGroup()
        self.joint_callback_group = ReentrantCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        
        # QoS profile for durability
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        
        # 图像订阅器 - 使用独立的回调组
        self.head_image_subscriber = self.create_subscription(
            CompressedImage,
            '/hdas/camera_head/left_raw/image_raw_color/compressed',
            self.head_image_callback,
            10,
            callback_group=self.image_callback_group
        )
        self.wrist_l_image_subscriber = self.create_subscription(
            CompressedImage,
            '/hdas/camera_wrist_left/left_raw/image_raw_color/compressed',
            self.wrist_l_image_callback,
            10,
            callback_group=self.image_callback_group
        )
        self.wrist_r_image_subscriber = self.create_subscription(
            CompressedImage,
            '/hdas/camera_wrist_right/right_raw/image_raw_color/compressed',
            self.wrist_r_image_callback,
            10,
            callback_group=self.image_callback_group
        )
        
        # 关节状态订阅器 - 使用独立的回调组
        self.arm_l_subscriber = self.create_subscription(
            JointState,
            '/hdas/feedback_arm_left',
            self.arm_l_callback,
            qos_profile,
            callback_group=self.joint_callback_group
        )
        self.gripper_l_subscriber = self.create_subscription(
            JointState,
            '/hdas/feedback_gripper_left',
            self.gripper_l_callback,
            qos_profile,
            callback_group=self.joint_callback_group
        )
        self.arm_r_subscriber = self.create_subscription(
            JointState,
            '/hdas/feedback_arm_right',
            self.arm_r_callback,
            qos_profile,
            callback_group=self.joint_callback_group
        )
        self.gripper_r_subscriber = self.create_subscription(
            JointState,
            '/hdas/feedback_gripper_right',
            self.gripper_r_callback,
            qos_profile,
            callback_group=self.joint_callback_group
        )
        
        # 控制发布器
        self.arm_l_publisher = self.create_publisher(
            JointState,
            '/motion_control/control_arm_left',
            qos_profile
        )
        self.gripper_l_publisher = self.create_publisher(
            JointState,
            '/motion_control/control_gripper_left',
            qos_profile
        )
        self.arm_r_publisher = self.create_publisher(
            JointState,
            '/motion_control/control_arm_right',
            qos_profile
        )
        self.gripper_r_publisher = self.create_publisher(
            JointState,
            '/motion_control/control_gripper_right',
            qos_profile
        )
        
        # 初始化数据存储
        self.cam_left = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cam_high = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.cam_right = np.zeros((480, 640, 3), dtype=np.uint8)
        self.arm_l_state = np.zeros(8, dtype=np.float32)
        self.arm_r_state = np.zeros(8, dtype=np.float32)
        
        # 线程锁，保护共享数据
        self.image_lock = threading.Lock()
        self.joint_lock = threading.Lock()
        
        # 性能监控
        self.image_count = 0
        self.joint_count = 0
        
        # 创建定时器用于状态监控
        self.status_timer = self.create_timer(
            5.0, 
            self.print_status,
            callback_group=self.control_callback_group
        )
        camera_matrix = np.array([
            [1006.0582118247934, 0.0, 960.5697664900398],
            [0.0, 1003.6652333957827, 553.0467321454616],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        distortion_coefficients = np.array([
            -0.38154841832755654, 0.19326013419893498, -0.002474956309358131, -0.000250071109775985, -0.0581107488155821
        ], dtype=np.float32)
        self.rectify_map = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_coefficients, None, camera_matrix, (1920, 1080), cv2.CV_32FC1
        )
        
        self.get_logger().info("多线程机器人节点启动完成")
    
    def distortion_correction(self, image):
        """对图像进行畸变校正"""
        try:
            if image is None or self.rectify_map is None:
                raise ValueError("图像或校正映射不能为空")
            # 使用校正映射进行畸变校正
            image = cv2.remap(image, self.rectify_map[0], self.rectify_map[1], cv2.INTER_LINEAR)
        except Exception as e:
            print(f"畸变校正失败: {str(e)}")
            return image
        return image

    def head_image_callback(self, msg):
        with self.image_lock:
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.cam_high = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                self.image_count += 1
                if self.cam_high is None:
                    self.get_logger().error("Failed to decode head image")
            except Exception as e:
                self.get_logger().error(f"Head image callback error: {e}")

    def wrist_l_image_callback(self, msg):
        with self.image_lock:
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.cam_left = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                self.image_count += 1
                if self.cam_left is None:
                    self.get_logger().error("Failed to decode left wrist image")
            except Exception as e:
                self.get_logger().error(f"Left wrist image callback error: {e}")

    def wrist_r_image_callback(self, msg):
        with self.image_lock:
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.cam_right = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                self.image_count += 1
                if self.cam_right is None:
                    self.get_logger().error("Failed to decode right wrist image")
            except Exception as e:
                self.get_logger().error(f"Right wrist image callback error: {e}")

    def arm_l_callback(self, msg):
        with self.joint_lock:
            try:
                if len(msg.position) >= 7:
                    self.arm_l_state[:7] = [msg.position[i].item() for i in range(7)]
                    self.joint_count += 1
                else:
                    self.get_logger().error("Received invalid arm left state")
            except Exception as e:
                self.get_logger().error(f"Arm left callback error: {e}")

    def gripper_l_callback(self, msg):
        with self.joint_lock:
            try:
                if len(msg.position) >= 1:
                    self.arm_l_state[7] = msg.position[0].item()
                    self.joint_count += 1
                else:
                    self.get_logger().error("Received invalid gripper left state")
            except Exception as e:
                self.get_logger().error(f"Gripper left callback error: {e}")

    def arm_r_callback(self, msg):
        with self.joint_lock:
            try:
                if len(msg.position) >= 7:
                    self.arm_r_state[:7] = [msg.position[i].item() for i in range(7)]
                    self.joint_count += 1
                else:
                    self.get_logger().error("Received invalid arm right state")
            except Exception as e:
                self.get_logger().error(f"Arm right callback error: {e}")

    def gripper_r_callback(self, msg):
        with self.joint_lock:
            try:
                if len(msg.position) >= 1:
                    self.arm_r_state[7] = msg.position[0].item()
                    self.joint_count += 1
                else:
                    self.get_logger().error("Received invalid gripper right state")
            except Exception as e:
                self.get_logger().error(f"Gripper right callback error: {e}")

    def arm_l_joint_control(self, joint_positions: np.ndarray):
        try:
            arm_msg = JointState()
            gripper_msg = JointState()
            
            timestamp = self.get_clock().now().to_msg()
            arm_msg.header.stamp = timestamp
            gripper_msg.header.stamp = timestamp
            
            arm_msg.position = [joint_positions[i].item() for i in range(7)]
            
            gripper_msg.position = [joint_positions[7].item()]
            
            self.arm_l_publisher.publish(arm_msg)
            self.gripper_l_publisher.publish(gripper_msg)
        except Exception as e:
            self.get_logger().error(f"Arm left control error: {e}")

    def arm_r_joint_control(self, joint_positions: np.ndarray):
        try:
            arm_msg = JointState()
            gripper_msg = JointState()
            
            timestamp = self.get_clock().now().to_msg()
            arm_msg.header.stamp = timestamp
            gripper_msg.header.stamp = timestamp

            arm_msg.position = [joint_positions[i].item() for i in range(7)]

            gripper_msg.position = [joint_positions[7].item()]
            
            self.arm_r_publisher.publish(arm_msg)
            self.gripper_r_publisher.publish(gripper_msg)
        except Exception as e:
            self.get_logger().error(f"Arm right control error: {e}")

    def get_head_image(self):
        with self.image_lock:
            """获取头部图像"""
            img = self.distortion_correction(self.cam_high.copy()) if self.cam_high is not None else None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_wrist_l_image(self):
        with self.image_lock:
            img = self.cam_left.copy() if self.cam_left is not None else None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_wrist_r_image(self):
        with self.image_lock:
            img = self.cam_right.copy() if self.cam_right is not None else None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_arm_l_state(self):
        with self.joint_lock:
            return self.arm_l_state.copy() if self.arm_l_state is not None else None

    def get_arm_r_state(self):
        with self.joint_lock:
            return self.arm_r_state.copy() if self.arm_r_state is not None else None

    def get_gripper_l_state(self):
        with self.joint_lock:
            return self.arm_l_state[7] if self.arm_l_state is not None else None

    def get_gripper_r_state(self):
        with self.joint_lock:
            return self.arm_r_state[7] if self.arm_r_state is not None else None

    def get_robot_joints_and_img(self):
        """获取机器人的关节状态和图像"""
        with self.image_lock, self.joint_lock:
            img1 = self.cam_left.copy() if self.cam_left is not None else None
            img2 = self.distortion_correction(self.cam_high.copy()) if self.cam_high is not None else None
            img3 = self.cam_right.copy() if self.cam_right is not None else None
            return {
                'head_image': cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
                'wrist_l_image': cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
                'wrist_r_image': cv2.cvtColor(img3, cv2.COLOR_BGR2RGB),
                'arm_l_state': self.arm_l_state.copy() if self.arm_l_state is not None else None,
                'arm_r_state': self.arm_r_state.copy() if self.arm_r_state is not None else None
            }
    
    def print_status(self):
        """定期打印状态信息"""
        self.get_logger().info(f"状态: 图像回调 {self.image_count}, 关节回调 {self.joint_count}")
        self.image_count = 0
        self.joint_count = 0

# 全局变量，用于管理executor和线程
_executor = None
_robot = None
_spin_thread = None
_running = False

def start_robot():
    """启动机器人节点（非阻塞）"""
    global _executor, _robot, _spin_thread, _running
    
    if _running:
        print("机器人节点已经在运行中")
        return _robot
    
    rclpy.init()
    
    try:
        _robot = Robot()
        _executor = MultiThreadedExecutor(num_threads=4)
        _executor.add_node(_robot)
        
        print("多线程机器人节点启动中...")
        print("线程分配:")
        print("  - 图像处理: 独立线程组")
        print("  - 关节状态: 独立线程组")
        print("  - 控制命令: 互斥线程组")
        
        # 在独立线程中运行executor
        _running = True
        _spin_thread = threading.Thread(target=_spin_executor, daemon=True)
        _spin_thread.start()
        
        print("机器人节点启动完成 (非阻塞模式)")
        return _robot
        
    except Exception as e:
        print(f"启动机器人节点失败: {e}")
        cleanup_robot()
        return None

def _spin_executor():
    """在独立线程中运行executor"""
    global _executor, _running
    
    try:
        while _running and rclpy.ok():
            _executor.spin_once(timeout_sec=0.1)
    except Exception as e:
        print(f"Executor 错误: {e}")
    finally:
        print("Executor 线程退出")

def stop_robot():
    """停止机器人节点"""
    global _running
    _running = False
    print("停止机器人节点...")

def cleanup_robot():
    """清理资源"""
    global _executor, _robot, _spin_thread, _running
    
    _running = False
    
    if _spin_thread and _spin_thread.is_alive():
        _spin_thread.join(timeout=2.0)
        if _spin_thread.is_alive():
            print("Spin线程未能正常结束")
    
    if _robot:
        _robot.destroy_node()
        _robot = None
    
    if _executor:
        _executor.shutdown()
        _executor = None
    
    try:
        rclpy.shutdown()
    except:
        pass
    
    print("资源清理完成")

def get_robot():
    """获取机器人实例"""
    global _robot
    return _robot

def is_robot_running():
    """检查机器人是否正在运行"""
    global _running
    return _running