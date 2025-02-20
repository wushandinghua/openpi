#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs
import numpy as np
reset_position = [0.08286023760904516, -0.3646991386298497, -3.062494372369459, -2.5787213066605315, 0.0279318077568611, -0.8443589520351491, 1.6482821971332426]
GRIPPER_WIDTH_MAX = 0.085
GRIPPER_JOINT_MAX, GRIPPER_JOINT_MIN = 0.79301, 0.00698

class KinovaControllerRos:
    def __init__(self, init_node=True):
        # init moveit 
        super(KinovaControllerRos, self).__init__()
        self.robot_name = "gen3"
        self.max_gripper_width = GRIPPER_WIDTH_MAX
        self.max_gripper_joint = GRIPPER_JOINT_MAX
        self.min_gripper_joint = GRIPPER_JOINT_MIN
        self.is_init_success = False
        self.relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.max_joint_delta = self.relative_max_joint_delta.max()
        
        try: 
            if init_node:
                rospy.init_node("kinova_controller_ros", anonymous=True)

            # setup gripper
            self._setup_gripper_params()
                        
            # init moveit_commander
            moveit_commander.roscpp_initialize(self.robot_name)
            
            # setup moveit interface
            self._setup_moveit_interface()
            
            rospy.loginfo("Successfully initialized node in namespace " + rospy.get_namespace())
            self.is_init_success = True
            
        except Exception as e:
            rospy.logerr(f"Initalization failed: {str(e)}")
            self.is_init_success = False
            return
        
    def _setup_gripper_params(self):
        """Setup gripper-related parameters"""
        self.is_gripper_present = rospy.get_param(
            rospy.get_namespace() + "is_gripper_present", False
        )
        rospy.loginfo("Is gripper present: %s", self.is_gripper_present)

        if self.is_gripper_present:
            gripper_joint_names = rospy.get_param(
                rospy.get_namespace() + "gripper_joint_names", []
            )
            self.gripper_joint_name = (
                gripper_joint_names[0] if gripper_joint_names else ""
            )
            rospy.loginfo("Gripper joint name: %s", self.gripper_joint_name)
        else:
            self.gripper_joint_name = ""

        self.degrees_of_freedom = rospy.get_param(
            rospy.get_namespace() + "degrees_of_freedom", 7
        )
    
    def _setup_moveit_interface(self):
        """Setup MoveIt interface components"""
        arm_group_name = "arm"
        self.robot = moveit_commander.RobotCommander("robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
        self.arm_group = moveit_commander.MoveGroupCommander(
            arm_group_name, ns=rospy.get_namespace()
        )

        # Setup display trajectory publisher
        self.display_trajectory_publisher = rospy.Publisher(
            rospy.get_namespace() + "move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        if self.is_gripper_present:
            gripper_group_name = "gripper"
            self.gripper_group = moveit_commander.MoveGroupCommander(
                gripper_group_name, ns=rospy.get_namespace()
            )

    def get_joint_states(self):
        return self.arm_group.get_current_joint_values()    

    def get_gripper_state(self):
        if self.is_gripper_present:
            print(self.gripper_group.get_current_joint_values())
            return (1-(abs(self.gripper_group.get_current_joint_values()[-1])-self.min_gripper_joint) / (self.max_gripper_joint-self.min_gripper_joint)) * self.max_gripper_width
        else:
            rospy.logwarn("Gripper is not present.")
            return None 
           
    def get_current_state(self):
        return self.get_joint_states(), self.get_gripper_state()

    def cartesian_control(self, target_pose):
        """
        cartesian control
        """
        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        return success
    
    def set_joint_positions(self, joint_positions, speed_limit=0.6):
        '''
        joint control
        '''
        self.arm_group.set_max_velocity_scaling_factor(speed_limit)
        
        try:
            self.arm_group.set_joint_value_target(joint_positions)
            
        except Exception as e:
            rospy.logerr(f"[move_to_joint]Failed to move to joint positions: {str(e)}")
            return False

        success = self.arm_group.go(wait=True)
        
        return success

    def set_joint_velocities(self, joint_velocities, speed_limit=0.6):
        '''
        joint velocity control
        '''
        joint_delta = self.joint_velocity_to_delta(joint_velocities)
        print('joint_delta:', joint_delta)
        joint_positions = list(joint_delta + np.array(self.get_joint_states()))
        print('joint_position:', joint_positions)
        return self.set_joint_positions(joint_positions, speed_limit=speed_limit)

    def set_gripper_width(self, relative_position, type = "meter"):
        """
        gripper control
        """
        if not self.is_gripper_present:
            rospy.logwarn("Gripper is not present.")
            return False

        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        
        if type == "joint":
            "夹爪电机位置"
            val = gripper_joint.move(relative_position, True)
        elif type == "100":
            "映射到0-100"
            val = gripper_joint.move(
                relative_position
                * (gripper_max_absolute_pos - gripper_min_absolute_pos)
                + gripper_min_absolute_pos,
                True,
            ) 
        elif type == "meter":
            joint = self.max_gripper_joint - (relative_position / self.max_gripper_width * (self.max_gripper_joint-self.min_gripper_joint))
            val = gripper_joint.move(joint,
                True,
            )

        return val
    
    def joint_velocity_to_delta(self, joint_velocity):
        if isinstance(joint_velocity, list):
            joint_velocity = np.array(joint_velocity)

        relative_max_joint_vel = self.joint_delta_to_velocity(self.relative_max_joint_delta)
        max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()

        if max_joint_vel_norm > 1:
            joint_velocity = joint_velocity / max_joint_vel_norm

        joint_delta = joint_velocity * self.max_joint_delta

        return joint_delta
    
    def joint_delta_to_velocity(self, joint_delta):
        if isinstance(joint_delta, list):
            joint_delta = np.array(joint_delta)

        return joint_delta / self.max_joint_delta

if __name__ == "__main__":
    arm = KinovaControllerRos()
    
    ## Example: Get state
    #rospy.sleep(1)
    #joints, gripper = arm.get_current_state()
    #print("current joint positions: {}".format(joints))
    #print("current gripper width: {:.3f}m".format(gripper))
    #
    ## Example: Move arm with joint positions
    #target_positions = [reset_position[i]+0.1 for i in range(len(reset_position))]
    #success = arm.set_joint_positions(target_positions)
    #if success:
    #    print("setting joint positions successed")
    #else:
    #    print("setting joint positions failed")

    ## # Example: Move arm with joint velocities
    #target_velocities = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #success = arm.set_joint_velocities(target_velocities)
    #if success:
    #    print("setting joint velocities successed")
    #else:
    #    print("setting joint velocities failed")

    ## # Example: Set gripper width
    #arm.set_gripper_width(0.04)
    ## print("current gripper width: {:.3f}m".format(gripper))
    rospy.spin()
