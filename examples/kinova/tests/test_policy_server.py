import numpy as np
from openpi_client import websocket_client_policy
import time

policy_client = websocket_client_policy.WebsocketClientPolicy(host="192.168.3.101", port=8000)

def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

def make_kinova_example() -> dict:
    """Creates a random input example for the Kinova policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

#example = droid_policy.make_droid_example()
example = make_kinova_example()
cnt=80
start_time = time.time()
for _ in range(cnt):
    action_chunk = policy_client.infer(example)["actions"]
    #print(action_chunk)
end_time = time.time()
print("耗时: {:.3f}秒".format((end_time - start_time)/cnt))
