from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
cam1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
cam2 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
cam3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
state = np.random.rand(16)
task_instruction = "Pick up the object and place it in the box."

num_steps = 1
for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "cam_left_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cam1, 224, 224)
        ),
        "cam_high": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cam2, 224, 224)
        ),
        "cam_right_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cam2, 224, 224)
        ),
        "state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    print("step:", step, "action_chunk:", action_chunk)