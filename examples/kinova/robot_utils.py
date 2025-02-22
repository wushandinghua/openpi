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
        self.qpos = [0.0] * 8

        if init_node:
            rospy.init_node("pi0", anonymous=True)

        rospy.Subscriber("/pi0/joint_states", JointState, self.robot_state_cb)
        self.position_command_publisher = rospy.Publisher("/pi0/joint_position_command", JointState, queue_size=1)
        self.velocity_command_publisher = rospy.Publisher("/pi0/joint_velocity_command", JointState, queue_size=1)
        time.sleep(0.5)

    def robot_state_cb(self, data):
        self.qpos = list(data.position)
        self.qvel = list(data.velocity)
        self.effort = list(data.effort)
        self.data = data

    def set_joint_positions(self, positions):
        data = JointState()
        data.position = positions
        self.position_command_publisher.publish(data)
    
    def set_joint_velocities(self, velocities):
        data = JointState()
        data.velocity = velocities
        self.velocity_command_publisher.publish(data)

#if __name__ == "__main__":
#    robot = Kinova()
#    time.sleep(0.5)
#    states = robot.qpos
#    print('current state:', states)
#
#    # Example: Move arm with joint positions
#    target_positions = [-0.05114174767722801, -0.32670062356438123, -3.067645192233781, -1.9892705467878438, 0.01766729831947718, -1.091837990020724, 1.7992638567067234, 0.04724378143654149]
#    robot.set_joint_positions(target_positions)
#    print("setting position finished")
#    time.sleep(5)
#
#    # Example: Move arm with joint velocities
#    target_velocities = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06]
#    robot.set_joint_velocities(target_velocities)
#    print("setting velocities finished")
#    time.sleep(0.5)
#    states = robot.qpos
#    print('current state:', states)
#    time.sleep(1)
#
#    image_recorder = ImageRecorder(init_node=True)
#    images = image_recorder.get_images()
#    base_path = '/home/cuhk/quebinbin/vla/pi/openpi/examples/kinova'
#    import cv2 as cv
#    cv.imwrite(f'{base_path}/cam_exterior.png', images['cam_exterior'])
#    cv.imwrite(f'{base_path}/cam_wrist.png', images['cam_wrist'])
#    print(images)