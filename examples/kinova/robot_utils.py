#!/usr/bin/env python
# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import time
import os
import sys
sys.path.append("./")

from examples.kinova import constants
from sensor_msgs.msg import JointState
from realsense2_camera.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
import numpy as np
import rospy
import math
import cv2
from kortex_driver.msg import Base_JointSpeeds, JointSpeed, Finger, GripperMode, WaypointList, Waypoint, AngularWaypoint, ActionEvent, ActionNotification
from kortex_driver.srv import SendGripperCommandRequest, SendGripperCommand, ExecuteActionRequest, ExecuteAction, ValidateWaypointList, OnNotificationActionTopicRequest, OnNotificationActionTopic


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
            self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="rgb8"),
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
            rgb_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
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

class KinovaV2:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.robot_name = "my_gen3"
        self.qpos = [0.0] * 8
        self.max_gripper_width = constants.GRIPPER_WIDTH_MAX
        self.max_gripper_joint = constants.GRIPPER_JOINT_MAX
        self.min_gripper_joint = constants.GRIPPER_JOINT_MIN

        try:
            if init_node:
                rospy.init_node("pi0", anonymous=True)

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            rospy.Subscriber(f"/{self.robot_name}/joint_states", JointState, self.robot_state_cb)
            # self.position_command_publisher = rospy.Publisher("/pi0/joint_position_command", JointState, queue_size=1)

            execute_action_full_name = f"/{self.robot_name}/base/execute_action"
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)
            
            self.velocity_command_publisher = rospy.Publisher(f"/{self.robot_name}/in/joint_velocity", Base_JointSpeeds, queue_size=1)

            send_gripper_command_full_name = f"/{self.robot_name}/base/send_gripper_command"
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

            validate_waypoint_list_full_name = '/' + self.robot_name + '/base/validate_waypoint_list'
            rospy.wait_for_service(validate_waypoint_list_full_name)
            self.validate_waypoint_list = rospy.ServiceProxy(validate_waypoint_list_full_name, ValidateWaypointList)
            time.sleep(0.5)
            self.subscribe_to_a_robot_notification()
        except:
            self.is_init_success = False
            rospy.logerr("Failed to initialize KinovaV2")
        else:
            self.is_init_success = True
            rospy.loginfo("initialize KinovaV2 successed")
    
    def robot_state_cb(self, data):
        self.qpos = list(data.position)[:8]
        self.qpos[-1] = (1-(abs(self.qpos[-1])-self.min_gripper_joint) / (self.max_gripper_joint-self.min_gripper_joint)) * self.max_gripper_width
    
    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)
            
    def subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)
        return True

    def set_joint_positions(self, positions): 
        self.last_action_notif_type = None
        req = ExecuteActionRequest()

        trajectory = WaypointList()
        waypoint = Waypoint()
        angularWaypoint = AngularWaypoint()

        # Angles to send the arm to vertical position (all zeros)
        for position in positions[:-1]:
            angularWaypoint.angles.append(position * 180 / math.pi)

        # Each AngularWaypoint needs a duration and the global duration (from WaypointList) is disregarded. 
        # If you put something too small (for either global duration or AngularWaypoint duration), the trajectory will be rejected.
        angular_duration = 0
        angularWaypoint.duration = angular_duration

        # Initialize Waypoint and WaypointList
        waypoint.oneof_type_of_waypoint.angular_waypoint.append(angularWaypoint)
        trajectory.duration = 0
        trajectory.use_optimal_blending = False
        trajectory.waypoints.append(waypoint)

        try:
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        error_number = len(res.output.trajectory_error_report.trajectory_error_elements)
        MAX_ANGULAR_DURATION = 30

        while (error_number >= 1 and angular_duration != MAX_ANGULAR_DURATION) :   
            angular_duration += 1
            trajectory.waypoints[0].oneof_type_of_waypoint.angular_waypoint[0].duration = angular_duration

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False
            error_number = len(res.output.trajectory_error_report.trajectory_error_elements)

        if (angular_duration == MAX_ANGULAR_DURATION) :
            # It should be possible to reach position within 30s
            # WaypointList is invalid (other error than angularWaypoint duration)
            rospy.loginfo("WaypointList is invalid")
            return False
        
        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)
        
        # Send the angles
        self.set_gripper_joint(positions[-1])
        try:
            self.execute_action(req)   
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteWaypointjectory")
            return False
        else:
            return self.wait_for_action_end_or_abort()
    
    def set_gripper_joint(self, value):
        # Initialize the request
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service 
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(0.5)
            return True
    
    def set_joint_velocities(self, velocities):
        data = Base_JointSpeeds()
        joint_speeds = []
        for i, velocity in enumerate(velocities[:-1]):
            joint_speed = JointSpeed()
            joint_speed.joint_identifier = i
            joint_speed.value = velocity
            joint_speed.duration = 0
            joint_speeds.append(joint_speed)
        data.joint_speeds = joint_speeds
        data.duration = 0
        self.velocity_command_publisher.publish(data)
        # stop the robot
        time.sleep(0.2)
        for joint_speed in data.joint_speeds:
            joint_speed.value = 0
        self.velocity_command_publisher.publish(data)
        gripper_width_relative = velocities[-1]
        # gripper_joint = self.max_gripper_joint - gripper_width / self.max_gripper_width * (self.max_gripper_joint-self.min_gripper_joint)
        self.set_gripper_joint(gripper_width_relative)

# if __name__ == "__main__":
#    robot = KinovaV2()
#    time.sleep(0.5)
#    states = robot.qpos
#    print('current state:', states)

#    # Example: Move arm with joint positions
#    target_positions = [-0.05114174767722801, -0.32670062356438123, -3.067645192233781, -1.9892705467878438, 0.01766729831947718, -1.091837990020724, 1.7992638567067234, 0.5]
#    robot.set_joint_positions(target_positions)
#    print("setting position finished")
#    time.sleep(5)

   # Example: Move arm with joint velocities
#    target_velocities = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06]
#    robot.set_joint_velocities(target_velocities)
#    print("setting velocities finished")
#    time.sleep(0.5)
#    states = robot.qpos
#    print('current state:', states)
#    time.sleep(1)

#    #robot = Kinova()
#    #time.sleep(0.5)
#    #states = robot.qpos
#    #print('current state:', states)
#
#    ## Example: Move arm with joint positions
#    #target_positions = [-0.05114174767722801, -0.32670062356438123, -3.067645192233781, -1.9892705467878438, 0.01766729831947718, -1.091837990020724, 1.7992638567067234, 0.04724378143654149]
#    #robot.set_joint_positions(target_positions)
#    #print("setting position finished")
#    #time.sleep(5)
#
#    ## Example: Move arm with joint velocities
#    #target_velocities = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06]
#    #robot.set_joint_velocities(target_velocities)
#    #print("setting velocities finished")
#    #time.sleep(0.5)
#    #states = robot.qpos
#    #print('current state:', states)
#    #time.sleep(1)
#
#    image_recorder = ImageRecorder(init_node=True)
#    images = image_recorder.get_images()
#    base_path = '/home/cuhk/quebinbin/vla/pi/openpi/examples/kinova'
#    import cv2 as cv
#    cv.imwrite(f'{base_path}/cam_exterior.png', images['cam_exterior'])
#    cv.imwrite(f'{base_path}/cam_wrist.png', images['cam_wrist'])
#    print(images)