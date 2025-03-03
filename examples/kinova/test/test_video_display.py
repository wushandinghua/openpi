import numpy as np
import sys
sys.path.append("./")
from examples.kinova import video_display as _video_display

# 测试 observation 数据格式
im = np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8)  # [C, H, W]
im1 = np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8)  # [C, H, W]
observation = {
    "camera": im,  # 假设 subscriber_name 为 "camera"
    "camera1": im1,  # 假设 subscriber_name 为 "camera"
    "observation/joint_position": [0.1, 0.2, 0.3],
    "observation/gripper_position": 0.5
}

display = _video_display.VideoDisplay(subscriber_name="camera")
display.on_episode_start()
display1 = _video_display.VideoDisplay(subscriber_name="camera1")
display1.on_episode_start()


display.on_step(observation, {})
display1.on_step(observation, {})
# 应显示随机生成的图像，标题包含关节状态。
import time
time.sleep(10)
display.on_episode_end()
display1.on_episode_end()
