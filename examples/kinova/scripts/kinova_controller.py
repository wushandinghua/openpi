#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from kortex_driver.srv import Base_GetMeasuredJointAngles, Base_GetMeasuredJointAnglesRequest
from kortex_driver.msg import Base_JointSpeeds, JointSpeed, GripperCommand, Base_JointPositions

class KinovaController:
    def __init__(self, init_node):
        self.robot_name = ""
        if init_node:
            rospy.init_node("kinova_controller", anonymous=True)
        # Service client for getting joint angles
        rospy.wait_for_service(f'/{self.robot_name}/base/get_measured_joint_angles')
        self.get_joint_angles_service = rospy.ServiceProxy(f'/{self.robot_name}/base/get_measured_joint_angles', Base_GetMeasuredJointAngles)
        
        # Publisher for joint speed control
        self.joint_speed_pub = rospy.Publisher(f'/{self.robot_name}/base/joint_speeds', Base_JointSpeeds, queue_size=10)

        # Publisher for joint position control
        self.joint_position_pub = rospy.Publisher(f'/{self.robot_name}/base/joint_positions', Base_JointPositions, queue_size=10)
        
        # Publisher for gripper control
        self.gripper_pub = rospy.Publisher(f'/{self.robot_name}/base/send_gripper_command', GripperCommand, queue_size=10)
        
    def get_joint_positions(self):
        try:
            response = self.get_joint_angles_service(Base_GetMeasuredJointAnglesRequest())
            joint_angles = [joint_angle.value for joint_angle in response.joint_angles.joint_angles]
            return joint_angles
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
    
    def set_joint_speeds(self, speeds):
        """Set the speed for each joint"""
        msg = Base_JointSpeeds()
        msg.joint_speeds = [JointSpeed(joint_identifier=i, value=speed, duration=0) for i, speed in enumerate(speeds)]
        self.joint_speed_pub.publish(msg)
    
    def set_joint_positions(self, positions):
        """Set the position for each joint"""

    def set_gripper_width(self, width):
        """Control the gripper width (0.0 to 1.0)"""
        msg = GripperCommand()
        msg.mode = GripperCommand.GRIPPER_POSITION
        msg.gripper.finger.append(width)
        self.gripper_pub.publish(msg)

if __name__ == "__main__":
    controller = KinovaController()
    
    # Example usage
    rospy.sleep(1)
    angles = controller.get_joint_positions()
    if angles:
        rospy.loginfo(f"Current joint angles: {angles}")
    
    # Example: Move joints with speed
    controller.set_joint_speeds([10, 10, 10, 10, 10, 10, 10])
    rospy.sleep(2)
    controller.set_joint_speeds([0, 0, 0, 0, 0, 0, 0])  # Stop movement
    
    # Example: Set gripper width
    controller.set_gripper_width(0.5)
    rospy.sleep(2)
    controller.set_gripper_width(0.0)
