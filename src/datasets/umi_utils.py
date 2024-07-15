import numpy as np
import collections
import os
import json
from tqdm import tqdm

def umi_read_poses_from_folder(data_save_dir, focal_length):
    subfolders = sorted(list(os.listdir(data_save_dir)))
    all_poses = {}
    for i, subfolder in tqdm(enumerate(subfolders)):
        subfolder_path = os.path.join(data_save_dir, subfolder)

        file = os.path.join(subfolder_path, 'raw_labels.json')
        data = json.load(open(file, "r"))
        dct = collections.defaultdict(list)
        frame_names = sorted(list(data.keys()))
        for frame in frame_names:
            pose = data[frame]
            pose = np.asarray(pose)
            dct['imgs'].append("images/" + frame)
            dct['poses'].append(pose)
        
        gripper_state_json = os.path.join(subfolder_path, 'gripper_state.json')
        gripper_state_data = json.load(open(gripper_state_json, "r"))
        gripper_state = []
        for img_path in dct['imgs']:
            img_pathkey = img_path.split("/")[-1]
            assert img_pathkey in gripper_state_data, f"{img_pathkey} not in {subfolder_path}/gripper_state.json"
            gripper_state.append(int(gripper_state_data[img_pathkey]))
        dct['gripper_state'] = gripper_state
        dct['sample_param'] = (1,15)
        dct['focal_length'] = float(focal_length)
        dct['has_gripper_state'] = True
        all_poses[subfolder_path] = dct
    
    return all_poses
