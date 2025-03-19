
# uv venv --python 3.10 examples/kinova/.venv
source examples/kinova/.venv/bin/activate
source ~/IS_Table/Kinova_End2End_Basic/ros_ws/devel/setup.zsh
# uv pip install -e packages/openpi-client

python examples/kinova/main.py --args.host "192.168.3.101" --args.port 8000 --args.instruction "Pick up the orange and put it in the drawer."