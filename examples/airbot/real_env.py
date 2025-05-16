import collections
import time
import copy
from typing import Optional, List
import dm_env
import numpy as np

from examples.airbot import constants
from examples.airbot import airbot_robot


class RealEnv:
    """
    Environment for kinova and robotiq
    Action space:      [arm_qpos (7),             # joint velocity
                        gripper_positions (1),]    # gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (7),          # absolute joint position
                                        gripper_position (1),] # gripper position (0: close, 1: open)
                                        
                        "images": {"exterior_image_1_left": (480x640x3),        # h, w, c, dtype='uint8'
                                   "wrist_image_left": (480x640x3),         # h, w, c, dtype='uint8'
                        }
    """

    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = reset_position[:7] if reset_position else constants.DEFAULT_RESET_POSITION

        # new kinova controller
        self.robot = airbot_robot.AIRBOTPlay()

        if setup_robots:
            self.setup_robots()

    def setup_robots(self):
        # reboot robot
        command = "reboot robot"
        self.robot.back_home()
        print(f"real env setup cmd:{command}, sleep time:{constants.DT}")

    def get_observation(self):
        # obs = collections.OrderedDict()   
        # obs["qpos"] = self.get_qpos()
        # obs["images"] = self.get_images()
        # return obs
        return self.robot.capture_observation()
        

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
        assert action.shape[-1] == 14
        self.robot.send_action(action)
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)