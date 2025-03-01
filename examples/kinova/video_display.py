import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoDisplay(_subscriber.Subscriber):
    """Displays video frames."""

    def __init__(self, subscriber_name_list) -> None:
        self._ax_wrist: plt.Axes | None = None
        self._plt_img_wrist: plt.Image | None = None
        self._ax_exterior: plt.Axes | None = None
        self._plt_img_exterior: plt.Image | None = None
        self._subscriber_name_list = subscriber_name_list
        self._ax_list = None
        self._plt_img_list = None

    @override
    def on_episode_start(self) -> None:
        plt.ion()
        #self._ax_wrist = plt.subplot(1,2,1)
        #self._ax_exterior = plt.subplot(1,2,2)
        #self._plt_img_wrist = None
        #self._plt_img_exterior = None
        self._ax_list = [plt.subplot(1, len(self._subscriber_name_list), i+1) for i in range(len(self._subscriber_name_list))]
        self._plt_img_list = [None for i in range(len(self._subscriber_name_list))]

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        assert self._ax_list is not None

        #im = observation["image"][0]  # [C, H, W]
        #im = observation[f"{self._subscriber_name}"]  # [C, H, W]
        #im = np.transpose(im, (1, 2, 0))  # [H, W, C]

        #im = observation["cam_wrist"]  # [C, H, W]
        #if self._plt_img_wrist is None:
        #    self._plt_img_wrist = self._ax_wrist.imshow(im)
        #else:
        #    self._plt_img.set_data(im)

        #self._ax.set_title(f'joint states:{observation["observation/joint_position"]},{observation["observation/gripper_position"]}')
        for idx in range(len(self._subscriber_name_list)):
            im = observation[f"observation/{self._subscriber_name_list[idx]}"]
            if self._plt_img_list[idx] is None:
                self._plt_img_list[idx] = self._ax_list[idx].imshow(im)
            else:
                self._plt_img_list[idx].set_data(im)
        plt.suptitle(f'joint states:{observation["observation/joint_position"]},{observation["observation/gripper_position"]}')
        plt.pause(0.001)

    @override
    def on_episode_end(self) -> None:
        plt.ioff()
        plt.close()