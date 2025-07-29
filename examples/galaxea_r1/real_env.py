import collections
import time
import copy
from typing import Optional, List
import dm_env
import numpy as np

from examples.airbot import constants
from examples.galaxea_r1 import robot
from examples.airbot.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST


class RealEnv:
    """
    Environment for galaxea robot.
    Action space:      [left_arm_qpos (7),             # joint position
                        left_gripper_positions (1),    # gripper position
                        right_arm_qpos (7),            
                        right_gripper_positions (1)]

    Observation space: {"state": [left_arm_qpos (7),             # joint position
                            left_gripper_positions (1),    # gripper position
                            right_arm_qpos (7),            
                            right_gripper_positions (1)]
                                        
                        "images": {"cam1": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam2": (1080x1920x3),         # h, w, c, dtype='uint8'
                                   "cam3": (480x640x3),         # h, w, c, dtype='uint8'
                        }
    """

    def __init__(self, reset_position: Optional[List[float]] = None, reset_type: int = 3):
        self._reset_position = reset_position[:7] if reset_position else constants.DEFAULT_RESET_POSITION
        self._reset_type = reset_type

        # new galaxea controller
        self.robot = robot.start_robot()

    def setup_robots(self):
        # reboot robot
        """
        command = "reboot robot"
        action = [0.663, 0.786, 0.263, -1.814, 1.042, -0.342, -0.305,  77.13088 ,  
                  0.663, -0.786, -0.263, -1.813, -1.042, -0.342, 0.305, 77.408394]
        self.robot.arm_l_joint_control(np.array(action[:8], dtype=np.float32))
        self.robot.arm_r_joint_control(np.array(action[8:16], dtype=np.float32))
        print("****" * 10)
        time.sleep(5)
        print("****" * 10)
        print(f"real env setup cmd:{command}, sleep time:{constants.DT}")
        """
        self.robot.init_torso_and_arms()

    def get_observation(self):
        obs = self.robot.get_robot_joints_and_img()
        ret = {}
        ret["observation.state"] = np.concatenate([obs["arm_l_state"], obs["arm_r_state"]])
        ret[f"observation.images.{CAM_HIGH}"] = obs["head_image"]
        ret[f"observation.images.{CAM_LEFT_WRIST}"] = obs["wrist_l_image"]
        ret[f"observation.images.{CAM_RIGHT_WRIST}"] = obs["wrist_r_image"]
        return ret
        

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        if not fake:
            # Reboot robot 
            self.setup_robots()
            print("real env setup finished")
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        assert action.shape[-1] == 16
        self.robot.arm_l_joint_control(action[:8])
        self.robot.arm_r_joint_control(action[8:16])
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def make_real_env(reset_position: Optional[List[float]] = None, reset_type: int = 3) -> RealEnv:
    return RealEnv(reset_position=reset_position, reset_type=reset_type)
