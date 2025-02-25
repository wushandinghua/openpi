import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoDisplay(_subscriber.Subscriber):
    """Displays video frames."""

    def __init__(self, subscriber_name) -> None:
        self._ax: plt.Axes | None = None
        self._plt_img: plt.Image | None = None
        self._subscriber_name = subscriber_name

    @override
    def on_episode_start(self) -> None:
        plt.ion()
        self._ax = plt.subplot()
        self._plt_img = None

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        assert self._ax is not None

        #im = observation["image"][0]  # [C, H, W]
        im = observation[f"{self._subscriber_name}"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]

        if self._plt_img is None:
            self._plt_img = self._ax.imshow(im)
        else:
            self._plt_img.set_data(im)

        self._ax.set_title(f'joint states:{observation["observation/joint_position"]},{observation["observation/gripper_position"]}')
        plt.pause(0.001)

    @override
    def on_episode_end(self) -> None:
        plt.ioff()
        plt.close()