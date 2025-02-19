#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from kortex_driver.srv import SetFingersPosition, SetFingersPositionRequest, SetJointPositions, SetJointPositionsRequest, SetJointVelocities, SetJointVelocitiesRequest
from kortex_driver.msg import FingerPosition

class KinovaController:
    def __init__(self, init_node=False):
        self.robot_name = ""
        self.joint_names = [
            'joint_1', 
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6',
            'joint_7'
        ]
        self.finger_joint_names = [
            'finger_joint',
            'finger_joint2'
        ]

        if init_node:
            rospy.init_node("kinova_controller", anonymous=True)
        
        # Subcriber for joint positions and gripper width
        self.current_joint_state = None
        rospy.Subscriber(f'/{self.robot_name}/joint_states', JointState, self.joint_state_callback)
        
        # init joint positions setting client
        rospy.wait_for_service(f'/{self.robot_name}/set_joint_positions')
        self.set_joint_positions_client = rospy.ServiceProxy(
            f'/{self.robot_name}/set_joint_positions',
            SetJointPositions
        )

        # init joint velocities setting client
        rospy.wait_for_service(f'/{self.robot_name}/set_joint_velocities')
        self.set_joint_velocities_client = rospy.ServiceProxy(
            f'/{self.robot_name}/set_joint_velocities',
            SetJointVelocities
        )
        
        # init gripper setting client
        rospy.wait_for_service(f'/{self.robot_name}/fingers_action/finger_positions')
        self.gripper_client = rospy.ServiceProxy(
            f'/{self.robot_name}/fingers_action/finger_positions',
            SetFingersPosition
        )
        
    def joint_state_callback(self, msg):
        """record latest state"""
        self.current_joint_state = msg

    def get_current_state(self):
        if self.current_joint_state is None:
            rospy.logwarn("haven't get joint state")
            return None, None

        try:
            # get joint positions
            joint_positions = []
            for name in self.joint_names:
                idx = self.current_joint_state.name.index(name)
                joint_positions.append(self.current_joint_state.position[idx])

            # get gripper width
            finger_width = 0.0
            for name in self.finger_joint_names:
                idx = self.current_joint_state.name.index(name)
                finger_width += self.current_joint_state.position[idx]

            return joint_positions, finger_width
        except ValueError as e:
            rospy.logerr("joint name miss match: {}".format(e))
            return None, None
    
    def set_joint_positions(self, positions, move_time=5.0):
        try:
            # create a service request
            req = SetJointPositionsRequest()
            req.positions = positions
            req.move_time = move_time

            # call service
            resp = self.set_joint_positions_client(req)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("setting joint positions failed: {}".format(e))
            return False

    def set_joint_velocities(self, velocities, duration=0.1):
        try:
            # Create a service request
            req = SetJointVelocitiesRequest()
            req.velocities = velocities
            req.duration = duration

            # call service
            resp = self.set_joint_velocities_client(req)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("setting joint velocities failed: {}".format(e))
            return False

    def set_gripper_width(self, width, tolerance=0.001):
        finger_position = width / 2.0

        # Create a service request
        req = SetFingersPositionRequest()
        req.fingers.finger1 = finger_position
        req.fingers.finger2 = finger_position

        try:
            resp = self.gripper_client(req)
            return resp.result == FingerPosition.SUCCESS
        except rospy.ServiceException as e:
            rospy.logerr("setting gripper width failed: {}".format(e))
            return False

if __name__ == "__main__":
    arm = KinovaController()
    
    # Example: Get state
    rospy.sleep(1)
    joints, gripper = arm.get_current_state()
    print("current joint positions: {}".format(joints))
    print("current gripper width: {:.3f}m".format(gripper))
    
    # Example: Move arm with joint positions
    target_positions = [0.0, 0.5, 0.0, 1.57, 0.0, 0.0, 0.0]
    success = arm.set_joint_positions(target_positions, move_time=3.0)
    if success:
        print("setting joint positions successed")
    else:
        print("setting joint positions failed")

    # Example: Move arm with joint velocities
    target_velocities = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    success = arm.set_joint_velocities(target_velocities, duration=0.5)
    if success:
        print("setting joint velocities successed")
    else:
        print("setting joint velocities failed")

    # Example: Set gripper width
    arm.set_gripper_width(0.04)
