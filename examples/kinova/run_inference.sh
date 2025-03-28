
# uv venv --python 3.10 examples/kinova/.venv
source examples/kinova/.venv/bin/activate
source ~/IS_Table/Kinova_End2End_Basic/ros_ws/devel/setup.zsh
uv pip install -e packages/openpi-client
uv pip install pyrealsense2
uv pip install pyqt5==5.13.0
wget https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl
uv pip install kortex_api-2.6.0.post3-py3-none-any.whl
uv pip install rospy

python examples/kinova/main.py --args.host "192.168.3.101" --args.port 8000 --args.instruction "Pick up the orange and put it in the drawer." --args.action_horizon 5