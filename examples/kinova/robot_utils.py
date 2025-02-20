#!/usr/bin/env python
# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import time
import os
import sys
sys.path.append("../../")

from examples.kinova import constants
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
            rospy.init_node("image_recorder", anonymous=True)
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

#if __name__ == "__main__":
#    image_recorder = ImageRecorder(init_node=True)
#    images = image_recorder.get_images()
#    base_path = '/home/cuhk/quebinbin/vla/pi/openpi/examples/kinova'
#    import cv2 as cv
#    cv.imwrite(f'{base_path}/cam_exterior.png', images['cam_exterior'])
#    cv.imwrite(f'{base_path}/cam_wrist.png', images['cam_wrist'])
#    print(images)
