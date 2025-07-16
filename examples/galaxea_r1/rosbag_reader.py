import argparse
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, JointState
import rosbag2_py
import glob
import os
import numpy as np
import cv2
import json
import time

def find_closest_index_ascending(sorted_arr, a):
    """在升序数组中查找最接近a的值的索引"""
    # 确定目标值a的插入位置（左侧插入点）
    idx = np.searchsorted(sorted_arr, a, side="left")

    # 边界处理
    if idx == 0:
        return 0
    if idx == len(sorted_arr):
        return len(sorted_arr) - 1

    # 比较左右相邻元素与a的差值
    left_diff = abs(a - sorted_arr[idx - 1])
    right_diff = abs(a - sorted_arr[idx])

    return idx - 1 if left_diff <= right_diff else idx

def read_bag(input_bag: str, param_file: str = None):
    """读取mcap文件中的消息"""
    bags = []
    if os.path.isdir(input_bag):
        bags = find_all_mcap_files(input_bag)
        if not bags:
            raise FileNotFoundError(f"No mcap files found in directory: {input_bag}")
    elif not input_bag.endswith(".mcap"):
        raise ValueError("Input path must be a directory containing mcap files or a single mcap file.")
    if not os.path.exists(input_bag):
        raise FileNotFoundError(f"Input bag file does not exist: {input_bag}")
    else:
        bags.append(input_bag)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    # filter topics based on the provided JSON file
    if param_file:
        if not os.path.exists(param_file):
            raise FileNotFoundError(f"Parameter JSON file does not exist: {param_file}")
        with open(param_file, "r") as f:
            params_map = json.load(f)
    else:
        raise ValueError("param_file must be provided to map topics to types.")
    reader.set_filter(
        rosbag2_py.StorageFilter(
            topics=[
                params_map["observations"]["images"]["cam1"],
                params_map["observations"]["images"]["cam2"],
                params_map["observations"]["images"]["cam3"],
                params_map["observations"]["state"]["arm_l_state"],
                params_map["observations"]["state"]["arm_r_state"],
                params_map["observations"]["state"]["gripper_l_state"],
                params_map["observations"]["state"]["gripper_r_state"],
                params_map["action"]["arm_l_action"],
                params_map["action"]["arm_r_action"],
                params_map["action"]["gripper_l_action"],
                params_map["action"]["gripper_r_action"],
            ]
        )
    )
    cam1 = []
    cam1_stamp = []
    cam2 = []
    cam2_stamp = []
    cam3 = []
    cam3_stamp = []
    arm_l_state = []
    arm_l_state_stamp = []
    arm_r_state = []
    arm_r_state_stamp = []
    gripper_l_state = []
    gripper_l_state_stamp = []
    gripper_r_state = []
    gripper_r_state_stamp = []
    arm_l_action = []
    arm_l_action_stamp = []
    arm_r_action = []
    arm_r_action_stamp = []
    gripper_l_action = []
    gripper_l_action_stamp = []
    gripper_r_action = []
    gripper_r_action_stamp = []

    topic_types = reader.get_all_topics_and_types()

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == params_map["observations"]["images"]["cam1"]:
            msg = deserialize_message(data, CompressedImage)
            cam1.append(msg.data)
            cam1_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["observations"]["images"]["cam2"]:
            msg = deserialize_message(data, CompressedImage)
            cam2.append(msg.data)
            cam2_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["observations"]["images"]["cam3"]:
            msg = deserialize_message(data, CompressedImage)
            cam3.append(msg.data)
            cam3_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["observations"]["state"]["arm_l_state"]:
            msg = deserialize_message(data, JointState)
            arm_l_state.append(np.array(msg.position[0:7]))
            arm_l_state_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["observations"]["state"]["arm_r_state"]:
            msg = deserialize_message(data, JointState)
            arm_r_state.append(np.array(msg.position[0:7]))
            arm_r_state_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["observations"]["state"]["gripper_l_state"]:
            msg = deserialize_message(data, JointState)
            gripper_l_state.append(msg.position[0])
            gripper_l_state_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["observations"]["state"]["gripper_r_state"]:
            msg = deserialize_message(data, JointState)
            gripper_r_state.append(msg.position[0])
            gripper_r_state_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["action"]["arm_l_action"]:
            msg = deserialize_message(data, JointState)
            arm_l_action.append(np.array(msg.position[0:7]))
            arm_l_action_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["action"]["arm_r_action"]:
            msg = deserialize_message(data, JointState)
            arm_r_action.append(np.array(msg.position[0:7]))
            arm_r_action_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["action"]["gripper_l_action"]:
            msg = deserialize_message(data, JointState)
            gripper_l_action.append(msg.position[0])
            gripper_l_action_stamp.append(int(msg.header.stamp.sec * 1e3))
        elif topic == params_map["action"]["gripper_r_action"]:
            msg = deserialize_message(data, JointState)
            gripper_r_action.append(msg.position[0])
            gripper_r_action_stamp.append(int(msg.header.stamp.sec * 1e3))
    del reader
    # Convert lists to numpy arrays
    arm_l_state = np.asarray(arm_l_state, dtype=np.float32)
    arm_r_state = np.asarray(arm_r_state, dtype=np.float32)
    gripper_l_state = np.asarray(gripper_l_state, dtype=np.float32)
    gripper_r_state = np.asarray(gripper_r_state, dtype=np.float32)
    arm_l_action = np.asarray(arm_l_action, dtype=np.float32)
    arm_r_action = np.asarray(arm_r_action, dtype=np.float32)
    gripper_l_action = np.asarray(gripper_l_action, dtype=np.float32)
    gripper_r_action = np.asarray(gripper_r_action)
    cam1_stamp = np.asarray(cam1_stamp, dtype=np.int64)
    cam2_stamp = np.asarray(cam2_stamp, dtype=np.int64)
    cam3_stamp = np.asarray(cam3_stamp, dtype=np.int64)
    arm_l_state_stamp = np.asarray(arm_l_state_stamp, dtype=np.int64)
    arm_r_state_stamp = np.asarray(arm_r_state_stamp, dtype=np.int64)
    gripper_l_state_stamp = np.asarray(gripper_l_state_stamp, dtype=np.int64)
    gripper_r_state_stamp = np.asarray(gripper_r_state_stamp, dtype=np.int64)
    arm_l_action_stamp = np.asarray(arm_l_action_stamp, dtype=np.int64)
    arm_r_action_stamp = np.asarray(arm_r_action_stamp, dtype=np.int64)
    gripper_l_action_stamp = np.asarray(gripper_l_action_stamp, dtype=np.int64)
    gripper_r_action_stamp = np.asarray(gripper_r_action_stamp, dtype=np.int64)
    first_timearry = np.asarray(
        [
            cam1_stamp[0],
            cam2_stamp[0],
            cam3_stamp[0],
            arm_l_state_stamp[0],
            arm_r_state_stamp[0],
            gripper_l_state_stamp[0],
            gripper_r_state_stamp[0],
            arm_l_action_stamp[0],
            arm_r_action_stamp[0],
            gripper_l_action_stamp[0],
            gripper_r_action_stamp[0],
        ]
    )
    # find the max time in all stamps first index
    max_time_index = np.argmax(first_timearry)
    max_time = first_timearry[max_time_index]

    first_index = find_closest_index_ascending(cam3_stamp, max_time)
    frame_num = len(cam3_stamp) - first_index
    print (f"Total frames: {frame_num}, Start index: {first_index}")
    vla_datas = {
        "observations": {
            "images": {
                "cam1": np.zeros((frame_num, 3, 480, 640), dtype=np.uint8),
                "cam2": np.zeros((frame_num, 3, 1080, 1920), dtype=np.uint8),
                "cam3": np.zeros((frame_num, 3, 480, 640), dtype=np.uint8),
            },
            "state": np.zeros((frame_num, 16), dtype=np.float32),
        },
        "action": np.zeros((frame_num, 16), dtype=np.float32),
    }
    camera_matrix = np.array(params_map["head_cam"]["camera_matrix"], dtype=np.float32)
    distortion_coefficients = np.array(
        params_map["head_cam"]["distortion_coefficients"], dtype=np.float32
    )
    head_rectify_map = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion_coefficients,
                None,  # 无需额外的旋转矩阵
                camera_matrix,  # 使用相同的相机矩阵
                (1920, 1080),  # 输出图像尺寸
                cv2.CV_32FC1,
            )
    np_idx = 0
    for time_idx in range(first_index, len(cam3_stamp)):
        time_now = cam3_stamp[time_idx]
        img = cv2.imdecode(np.frombuffer(cam3[time_idx], np.uint8), cv2.IMREAD_COLOR)
        vla_datas["observations"]["images"]["cam3"][np_idx] = np.transpose(img, (2, 0, 1))
        other_idx = find_closest_index_ascending(cam1_stamp, time_now)
        img = cv2.imdecode(np.frombuffer(cam1[other_idx], np.uint8), cv2.IMREAD_COLOR)
        vla_datas["observations"]["images"]["cam1"][np_idx] = np.transpose(img, (2, 0, 1))
        other_idx = find_closest_index_ascending(cam2_stamp, time_now)
        img = cv2.imdecode(np.frombuffer(cam2[other_idx], np.uint8), cv2.IMREAD_COLOR)
        img = distortion_correction(img, head_rectify_map)
        # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(f"test.jpg", img)
        vla_datas["observations"]["images"]["cam2"][np_idx] = np.transpose(img, (2, 0, 1))

        other_idx = find_closest_index_ascending(arm_l_state_stamp, time_now)
        vla_datas["observations"]["state"][np_idx, 0:7] = arm_l_state[other_idx]
        other_idx = find_closest_index_ascending(gripper_l_state_stamp, time_now)
        vla_datas["observations"]["state"][np_idx, 7:8] = gripper_l_state[other_idx]
        other_idx = find_closest_index_ascending(arm_r_state_stamp, time_now)
        vla_datas["observations"]["state"][np_idx, 8:15] = arm_r_state[other_idx]
        other_idx = find_closest_index_ascending(gripper_r_state_stamp, time_now)
        vla_datas["observations"]["state"][np_idx, 15:16] = gripper_r_state[other_idx]

        other_idx = find_closest_index_ascending(arm_l_action_stamp, time_now)
        vla_datas["action"][np_idx, 0:7] = arm_l_action[other_idx]
        other_idx = find_closest_index_ascending(gripper_l_action_stamp, time_now)
        vla_datas["action"][np_idx, 7:8] = gripper_l_action[other_idx]
        other_idx = find_closest_index_ascending(arm_r_action_stamp, time_now)
        vla_datas["action"][np_idx, 8:15] = arm_r_action[other_idx]
        other_idx = find_closest_index_ascending(gripper_r_action_stamp, time_now)
        vla_datas["action"][np_idx, 15:16] = gripper_r_action[other_idx]
        np_idx += 1
        print(f"Processing frame {np_idx}/{frame_num}", end="\r")
    return vla_datas

def find_all_mcap_files(input_path: str):
    return glob.glob(os.path.join(input_path, "**", "*.mcap"), recursive=True)


def distortion_correction(image, rectify_map):
    """对图像进行畸变校正"""
    try:
        
        # 使用校正映射进行畸变校正
        image = cv2.remap(image, rectify_map[0], rectify_map[1], cv2.INTER_LINEAR)
    except Exception as e:
        print(f"畸变校正失败: {str(e)}")
        return image
    return image


def cropped_image(image, crop_width, crop_height):
    """裁剪图像"""
    if image is not None:
        h, w = image.shape[:2]
        x_start = (w - crop_width) // 2
        y_start = (h - crop_height) // 2
        return image[y_start : y_start + crop_height, x_start : x_start + crop_width]
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ros_mcap_file_or_folder", help="input bag path (folder or filepath) to read from")
    parser.add_argument("param_json_file", help="input parameter JSON file path")

    args = parser.parse_args()
    bags = []
    print(f"Reading from: {args.ros_mcap_file_or_folder}")
    if os.path.isdir(args.ros_mcap_file_or_folder):
        bags = find_all_mcap_files(args.ros_mcap_file_or_folder)
        if not bags:
            raise FileNotFoundError(f"No mcap files found in directory: {args.ros_mcap_file_or_folder}")
    else:
        if not args.ros_mcap_file_or_folder.endswith(".mcap"):
            raise ValueError("Input path must be a directory containing mcap files or a single mcap file.")
        if not os.path.exists(args.ros_mcap_file_or_folder):
            raise FileNotFoundError(f"Input bag file does not exist: {args.ros_mcap_file_or_folder}")
        else:
            bags.append(args.ros_mcap_file_or_folder)
    for bag in bags:
        print(f"Reading bag file: {bag}")
        # 读取mcap文件
        dataset = read_bag(bag, args.param_json_file)


if __name__ == "__main__":
    '''
    dependencies: (ros2 humble has been tested)
    sudo apt install ros2-ros-humble-rosbag2-*
    usage: python3 rosbag_reader.py /path/to/rosbags_or_dir /path/to/params.json
    '''
    main()

