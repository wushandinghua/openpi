import os
import json
import shutil
from collections import OrderedDict

# 路径设置
ds_base = os.path.expanduser("~/.cache/huggingface/lerobot/qbb/open_close_tap_0709")
videos_dir = os.path.join(ds_base, "videos")
meta_dir = os.path.join(ds_base, "meta")
info_json_path = os.path.join(meta_dir, "info.json")

# 相机重命名映射
cam_map = {
    "observation.images.cam1": "observation.images.cam_left_wrist",
    "observation.images.cam2": "observation.images.cam_high",
    "observation.images.cam3": "observation.images.cam_right_wrist",
}

# 1. 重命名 videos 目录下的相机子文件夹
for chunk in os.listdir(videos_dir):
    chunk_path = os.path.join(videos_dir, chunk)
    if not os.path.isdir(chunk_path):
        continue
    for old, new in cam_map.items():
        old_path = os.path.join(chunk_path, old)
        new_path = os.path.join(chunk_path, new)
        if os.path.exists(old_path):
            print(f"Renaming {old_path} -> {new_path}")
            os.rename(old_path, new_path)

# 2. 修改 info.json
with open(info_json_path, "r") as f:
    info = json.load(f)

# features = info.get("features", {})

# 2.1 observation.state 和 action 的 names 处理
# for key in ["observation.state", "action"]:
#     if key in features and "names" in features[key]:
#         names = features[key]["names"]
#         if isinstance(names, list) and len(names) == 1 and len(names[0]) == 14:
#             # 交换前7维和后7维
#             features[key]["names"][0] = names[0][7:14] + names[0][0:7]

# 2.2 重命名 features 下的相机 key
with open(info_json_path, "r") as f:
    info = json.load(f, object_pairs_hook=OrderedDict)

features = info.get("features", OrderedDict())

# 只改名，不改变顺序
for old, new in cam_map.items():
    if old in features:
        items = list(features.items())
        features = OrderedDict(
            (new if k == old else k, v) for k, v in items
        )

info["features"] = features

# 保存
with open(info_json_path, "w") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)