from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import numpy as np

from examples.airbot import real_env as _real_env
from examples.airbot.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST


class AirbotEnvironment(_environment.Environment):
    """An environment for an kinova."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        instruction: str = None,
        reset_type: int = 3 # 0: no reset, 1: reset when robot connect, 2: reset when robot disconnect, 3: reset when robot connect and disconnect
    ) -> None:
        self._env = _real_env.make_real_env(reset_position=reset_position, reset_type=reset_type)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None
        self.instruction = instruction

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        # for k in list(obs["images"].keys()):
        #     if "_depth" in k:
        #         del obs["images"][k]

        #print("cam wrist image shape before:", np.array(obs["images"][f"{CAM_WRIST}"]).shape)
        #print("cam exterior image shape before:", np.array(obs["images"][f"{CAM_WRIST}"]).shape)
        # for cam_name in obs["images"]:
        #     #img = image_tools.convert_to_uint8(
        #     #    image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
        #     #)
        #     #obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
        #     img = image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
        #     obs["images"][cam_name] = img 
        for key in obs.keys():
            #img = image_tools.convert_to_uint8(
            #    image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            #)
            #obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            if "images" not in key: continue
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs[key], self._render_height, self._render_width)
            )
            obs[key] = einops.rearrange(img, "h w c -> c h w")

        #print("cam wrist image shape after:", obs["images"][f"{CAM_WRIST}"].shape)
        #print("cam exterior image shape after:", obs["images"][f"{CAM_WRIST}"].shape)
        # return {
        #     "state": obs["qpos"],
        #     "images": obs["images"],
        # }
        return {
            CAM_HIGH: obs[f"observation.images.{CAM_HIGH}"],
            CAM_LEFT_WRIST: obs[f"observation.images.{CAM_LEFT_WRIST}"],
            CAM_RIGHT_WRIST: obs[f"observation.images.{CAM_RIGHT_WRIST}"],
            "state": np.array(obs["observation.state"]),
            "prompt": self.instruction,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])