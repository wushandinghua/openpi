import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import sys
sys.path.append("./")
from examples.galaxea_r1 import env as _env
from examples.airbot import video_display as _video_display
from examples.airbot.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST
"""
cd /path/to/openpi

1. Install dependencies:
uv pip install typing_extensions==4.12.2
uv pip install tyro==0.9.5
uv pip install dm_env==1.6
uv pip install matplotlib==3.10.0
uv pip install matplotlib-inline==0.1.7
uv pip install -e packages/openpi-client

2.exapmle usage: 
python examples/galaxea_r1/main.py --args.host "192.168.3.101" --args.port 8000 --args.instruction "pick up a bottle and put it down into the box"
"""

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 10

    num_episodes: int = 1
    max_episode_steps: int = 1000
    instruction: str = "pick up the apple"
    reset_type: int = 3 # 0: no reset, 1: reset when robot connect, 2: reset when robot disconnect, 3: reset when robot connect and disconnect


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")
    cam_subscriber = _video_display.VideoDisplay([CAM_LEFT_WRIST, CAM_HIGH, CAM_RIGHT_WRIST])

    metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(
        environment=_env.GalaxeaEnvironment(reset_position=metadata.get("reset_pose"), instruction=args.instruction, reset_type=args.reset_type),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[cam_subscriber],
        #subscribers=[],
        max_hz=20,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)