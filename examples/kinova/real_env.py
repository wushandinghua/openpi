import collections
import time
import copy
from typing import Optional, List
import dm_env
import numpy as np

from examples.kinova import constants
from examples.kinova import robot_utils


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
        self.robot = robot_utils.Kinova(init_node)

        if setup_robots:
            self.setup_robots()

        self.image_recorder = robot_utils.ImageRecorder(init_node=False)

    def setup_robots(self):
        # reboot robot
        self.robot.set_joint_positions(self._reset_position)
        gripper_width = constants.GRIPPER_POSITION_UNNORMALIZE_FN(0)
        command = copy.deepcopy(self._reset_position)
        command.append(gripper_width)
        self.robot.set_joint_positions(command)
        time.sleep(constants.DT)
        time.sleep(15)
        print(f"real env setup cmd:{command}, sleep time:{constants.DT}")

    def get_qpos(self):
        positions = self.robot.qpos
        arm_qpos = positions[:7]
        gripper_qpos = [
            constants.GRIPPER_POSITION_NORMALIZE_FN(positions[7])
        ]
        return np.concatenate([arm_qpos, gripper_qpos])

    def get_images(self):
        images = self.image_recorder.get_images()
        #return self.image_recorder.get_images()
        return images

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["images"] = self.get_images()
        return obs

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
        # to match pi0-fast-droid model
        assert action.shape[-1] == 8
        joint_velocities = action[:-1]
        joint_velocities = np.clip(joint_velocities, -1, 1).tolist()
        gridder_width_relative = 1.0 if action[-1] > 0.5 else 0
        gripper_width = constants.GRIPPER_POSITION_UNNORMALIZE_FN(gridder_width_relative)
        joint_velocities.append(gripper_width)
        print("env.step cmd:", joint_velocities)
        self.robot.set_joint_velocities(joint_velocities)
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)