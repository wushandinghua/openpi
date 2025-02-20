#!/usr/bin/env python
# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import time
import os
import sys
sys.path.append("../../")

from examples.kinova import constants
from sensor_msgs.msg import JointState
from realsense2_camera.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
import numpy as np
import rospy
import cv2


class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = [constants.CAM_EXTERIOR,  constants.CAM_WRIST]

        if init_node:
            rospy.init_node("pi0", anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_depth_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)
            if cam_name == constants.CAM_EXTERIOR:
                callback_func = self.image_cb_cam_exterior
            elif cam_name == constants.CAM_WRIST:
                callback_func = self.image_cb_cam_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(
            self,
            f"{cam_name}_rgb_image",
            self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="bgr8"),
        )
        # setattr(
        #     self,
        #     f"{cam_name}_depth_image",
        #     self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
        # )
        setattr(
            self,
            f"{cam_name}_timestamp",
            data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
        )
        # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
        # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
        # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs + data.images[0].header.stamp.nsecs * 1e-9
            )

    def image_cb_cam_exterior(self, data):
        cam_name = constants.CAM_EXTERIOR
        return self.image_cb(cam_name, data)

    def image_cb_cam_wrist(self, data):
        cam_name = constants.CAM_WRIST
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            while getattr(self, f"{cam_name}_timestamp") <= self.cam_last_timestamps[cam_name]:
                time.sleep(0.00001)
            gbr_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
            # bgr to rgb
            rgb_image = cv2.cvtColor(gbr_image, cv2.COLOR_BGR2RGB)
            image_dict[cam_name] = rgb_image
            image_dict[f"{cam_name}_depth"] = depth_image
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
            print(f"{cam_name} {image_freq=:.2f}")
        print()

class Kinova:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug

        if init_node:
            rospy.init_node("pi0", anonymous=True)

        rospy.Subscriber(f"/kinova_controller_ros/joint_states", JointState, self.robot_state_cb)
        self.position_command_publisher = rospy.Publisher("position_joint_command", JointState, queue_size=1)
        self.velocity_command_publisher = rospy.Publisher("position_velocity_command", JointState, queue_size=1)

    def robot_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data

    def set_joint_positions(self, positions):
        data = JointState()
        data.positions = positions
        self.position_command_publisher.publish(data)
    
    def set_joint_velocities(self, velocities):
        data = JointState()
        data.velocity = velocities
        self.velocity_command_publisher.publish(data)

if __name__ == "__main__":
    robot = Kinova()
    states = robot.qpos
    print('current state:', states)

    # Example: Move arm with joint positions
    target_positions = [1.1717908796348647e-05, -0.35005974211768365, 
                        3.1400381664615136, -2.54007670871461, -7.084008499624872e-05, 
                        -0.8700355533690383, 1.5699754073888796, 0.06]
    robot.set_joint_positions(target_positions)
    print("setting position finished")

    # Example: Move arm with joint velocities
    target_velocities = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]
    robot.set_joint_velocities(target_velocities)
    print("setting velocities finished")

#    image_recorder = ImageRecorder(init_node=True)
#    images = image_recorder.get_images()
#    base_path = '/home/cuhk/quebinbin/vla/pi/openpi/examples/kinova'
#    import cv2 as cv
#    cv.imwrite(f'{base_path}/cam_exterior.png', images['cam_exterior'])
#    cv.imwrite(f'{base_path}/cam_wrist.png', images['cam_wrist'])
#    print(images)