import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import sys
sys.path.append("./")
from examples.airbot import env as _env
from examples.airbot import video_display as _video_display
from examples.airbot.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST
"""
1.conda activate airbot_5_8
2.
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
or 针对推断要用的包单独安装
uv pip install typing_extensions==4.12.2
uv pip install -e packages/openpi-client
uv pip install tyro==0.9.5
uv pip install dm_env==1.6
uv pip install matplotlib==3.10.0
uv pip install matplotlib-inline==0.1.7

3.exapmle usage: python examples/airbot/main.py --args.host "192.168.3.101" --args.port 8000 --args.instruction "pick up the banana with one of your arm and put it in the black fruit basket"
"""

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 10

    num_episodes: int = 1
    max_episode_steps: int = 1000
    instruction: str = "pick up the apple"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")
    cam_subscriber = _video_display.VideoDisplay([CAM_LEFT_WRIST, CAM_HIGH, CAM_RIGHT_WRIST])

    metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(
        environment=_env.AirbotEnvironment(reset_position=metadata.get("reset_pose"), instruction=args.instruction),
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