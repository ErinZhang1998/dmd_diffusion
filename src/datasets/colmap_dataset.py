import sys
import os

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_path)

import torch
from torch.utils import data

from PIL import Image
import numpy as np
import json
import sys
from utils import *
import torchvision
from write_translations import *
from slam_utils import read_pose
import umi_utils

class ColmapDataset(data.Dataset):
    def __init__(self,root_dirs, focal_lengths):
        super().__init__()
        assert len(root_dirs) == len(focal_lengths)
        
        self.poses = {}
        valid_num_data = 0
        for root_idx,root_dir in enumerate(root_dirs):
            focal_length = focal_lengths[root_idx]
            
            for f in os.listdir(root_dir):
                full_path = os.path.join(root_dir, f)
                if "." in f or os.path.exists(os.path.join(root_dir, f, "FAIL")):
                    continue 
                dct = get_colmap_labels(full_path, focal_length=focal_length, data_type="apple")
                if dct is None:
                    continue
                self.poses[full_path] = dct
                valid_num_data += 1
        print("valid_num_data: ", valid_num_data)
        print("focal_lengths: ", focal_lengths)

        self.sequence_codes = list(self.poses.keys())
        self.im_size = 128

    def __len__(self):
        return len(self.sequence_codes)

    def __getitem__(self,idx):
        folder_path = self.sequence_codes[idx]
        poses = self.poses[self.sequence_codes[idx]]
        available_idxs = list(range(len(poses['imgs'])))
        
        scale_factor = poses['scale_factor']
        # choose frames
        chosen_idxs = np.random.choice(available_idxs,2,replace=False)

        id_a = chosen_idxs[0]
        id_b = chosen_idxs[1]

        #frame_a = Image.open(self.root + self.sequence_codes[idx] + '/' + poses['imgs'][id_a])
        #frame_b = Image.open(self.root + self.sequence_codes[idx] + '/' + poses['imgs'][id_b])
        frame_a_path = os.path.join(folder_path, poses['imgs'][id_a])
        frame_b_path = os.path.join(folder_path, poses['imgs'][id_b])
        frame_a = Image.open(frame_a_path)
        frame_b = Image.open(frame_b_path)

        # ensure images have 360 height
        if frame_a.size[1] != 360:
            new_w = round(frame_a.size[0] * (360/frame_a.size[1]))
            frame_a = np.asarray(frame_a.resize((new_w,360)))
            frame_b = np.asarray(frame_b.resize((new_w,360)))

        # crop and downsample
        left_pos = (frame_a.shape[1]-360)//2
        frame_a_cropped = frame_a[:,left_pos:left_pos+360,:]
        frame_b_cropped = frame_b[:,left_pos:left_pos+360,:]
        im_a = Image.fromarray(frame_a_cropped).resize((256,256))
        im_b = Image.fromarray(frame_b_cropped).resize((256,256))
        im_a = np.asarray(im_a).transpose(2,0,1)/127.5 - 1
        im_b = np.asarray(im_b).transpose(2,0,1)/127.5 - 1

        # scale image values
        frame_a_cropped = frame_a_cropped/127.5 - 1
        frame_b_cropped = frame_b_cropped/127.5 - 1

        tform_a = poses['poses'][chosen_idxs[0]]
        tform_b = poses['poses'][chosen_idxs[1]]
        
        tform_a_inv = np.linalg.inv(tform_a)
        tform_b_inv = np.linalg.inv(tform_b)
        tform_ref = np.eye(4)
        tform_a_relative = np.matmul(tform_b_inv,tform_a)
        tform_b_relative = np.matmul(tform_a_inv,tform_b)
        
        tform_a_relative[:-1,-1] /= scale_factor
        tform_b_relative[:-1,-1] /= scale_factor

        focal_y_a = poses['focal_y']
        focal_y_b = poses['focal_y']

        camera_enc_ref = rel_camera_ray_encoding(tform_ref,self.im_size,focal_y_a)
        camera_enc_a = rel_camera_ray_encoding(tform_a_relative,self.im_size,focal_y_a)
        camera_enc_b = rel_camera_ray_encoding(tform_b_relative,self.im_size,focal_y_b)
        
        out_dict = {
            'sequence_code':self.sequence_codes[idx],
            'im_a': im_a.astype(np.float32),
            'im_b': im_b.astype(np.float32),
            'im_a_full': frame_a_cropped.transpose(2,0,1).astype(np.float32),
            'im_b_full': frame_b_cropped.transpose(2,0,1).astype(np.float32),
            'camera_enc_ref': camera_enc_ref,
            'camera_enc_a': camera_enc_a,
            'camera_enc_b': camera_enc_b,
            'tform_ref': tform_ref,
            'tform_a_relative': tform_a_relative,
            'tform_b_relative': tform_b_relative,
            'tform_ref': tform_ref,
            'tform_a': tform_a,
            'tform_b': tform_b,
            'focal_a': focal_y_a,
            'focal_b': focal_y_b,
        }
        return out_dict

class SlamDataset(data.Dataset):
    def __init__(self,root_dirs, focal_lengths):
        super().__init__()
        assert len(root_dirs) == len(focal_lengths)
        
        self.poses = {}
        valid_num_data = 0
        for root_idx,root_dir in enumerate(root_dirs):
            focal_length = focal_lengths[root_idx]
            for f in os.listdir(root_dir):
                full_path = os.path.join(root_dir, f)
                label_json_path = os.path.join(full_path, "raw_labels.json")
                if not os.path.exists(label_json_path):
                    print(f"Cannot find {label_json_path}")
                    continue 
                dct = read_pose(label_json_path)
                if dct is None or len(dct) == 0:
                    continue
                gripper_state_json = os.path.join(full_path, "gripper_state.json")
                has_gripper_state = False
                if os.path.exists(gripper_state_json):
                    gripper_state_data = json.load(open(gripper_state_json, "r"))
                    gripper_state = []
                    for img_path in dct['imgs']:
                        img_pathkey = img_path.split("/")[-1]
                        assert img_pathkey in gripper_state_data, f"{img_pathkey} not in {full_path}/gripper_state.json"
                        gripper_state.append(int(gripper_state_data[img_pathkey]))
                    dct['gripper_state'] = gripper_state
                    has_gripper_state = True
                
                dct['sample_param'] = (15,45) # (15,100) for play data
                dct['focal_length'] = float(focal_length)
                dct['has_gripper_state'] = has_gripper_state
                
                self.poses[full_path] = dct
                valid_num_data += 1
            
        print("valid_num_data: ", valid_num_data)
        print("focal_lengths: ", focal_lengths)
        
        self.sequence_codes = list(self.poses.keys())
        self.im_size = 128
        
        self.sample_distance = 15
        self.sample_range = 45

    def __len__(self):
        return len(self.sequence_codes)

    def __getitem__(self,idx):
        # idx indicates sequence idx, we will pick 2 random frames from there
        folder_path = self.sequence_codes[idx]
        poses = self.poses[folder_path]
        available_idxs = list(range(len(poses['imgs'])))
        sample_distance, sample_range = poses['sample_param']
        has_gripper_state = poses['has_gripper_state']
        # choose frames, make sure the chosen frames are at least sample_distance apart
        while True:
            id_a = np.random.choice(available_idxs,1,replace=False)[0]
            if has_gripper_state:
                a_gripper_state = poses['gripper_state'][id_a]
                if a_gripper_state < 0:
                    continue
            other_indices = []
            lower_ub = id_a-sample_distance
            if lower_ub >= 0:
                other_indices += list(range(max(0, lower_ub-sample_range), lower_ub+1))
            upper_lb = id_a+sample_distance
            if upper_lb <= len(poses['imgs'])-1:
                other_indices += list(range(upper_lb, min(len(poses['imgs']), upper_lb+sample_range)))
            if has_gripper_state:
                other_indices = [i for i in other_indices if poses['gripper_state'][i] == a_gripper_state]
                if len(other_indices) < 1:
                    continue
            id_b = np.random.choice(other_indices,1,replace=False)[0]
            
            break

        frame_a_path = os.path.join(folder_path, poses['imgs'][id_a])
        frame_b_path = os.path.join(folder_path, poses['imgs'][id_b])
        frame_a = Image.open(frame_a_path)
        frame_b = Image.open(frame_b_path)
        
        orig_w, orig_h = frame_a.size

        # ensure images have 360 height
        if orig_h != 360:
            new_w = round(frame_a.size[0] * (360/frame_a.size[1]))
            frame_a = np.asarray(frame_a.resize((new_w,360)))
            frame_b = np.asarray(frame_b.resize((new_w,360)))
        else:
            frame_a = np.asarray(frame_a)
            frame_b = np.asarray(frame_b)
        if orig_w != 360: # if the width is not 360
            # crop and downsample
            left_pos = (orig_w-360)//2
            frame_a_cropped = frame_a[:,left_pos:left_pos+360,:]
            frame_b_cropped = frame_b[:,left_pos:left_pos+360,:]
        else:
            frame_a_cropped = frame_a
            frame_b_cropped = frame_b
        im_a = Image.fromarray(frame_a_cropped).resize((256,256))
        im_b = Image.fromarray(frame_b_cropped).resize((256,256))
        im_a = np.asarray(im_a).transpose(2,0,1)/127.5 - 1
        im_b = np.asarray(im_b).transpose(2,0,1)/127.5 - 1

        # scale image values
        frame_a_cropped = frame_a_cropped/127.5 - 1
        frame_b_cropped = frame_b_cropped/127.5 - 1

        tform_a = poses['poses'][id_a]
        tform_b = poses['poses'][id_b]
        
        tform_a_inv = np.linalg.inv(tform_a)
        tform_b_inv = np.linalg.inv(tform_b)
        tform_ref = np.eye(4)
        tform_a_relative = np.matmul(tform_b_inv,tform_a)
        tform_b_relative = np.matmul(tform_a_inv,tform_b)
        
        focal_y_a = poses['focal_length'] / float(orig_h)
        focal_y_b = poses['focal_length'] / float(orig_h)

        camera_enc_ref = rel_camera_ray_encoding(tform_ref,self.im_size,focal_y_a)
        camera_enc_a = rel_camera_ray_encoding(tform_a_relative,self.im_size,focal_y_a)
        camera_enc_b = rel_camera_ray_encoding(tform_b_relative,self.im_size,focal_y_b)
        
        out_dict = {
            'sequence_code':self.sequence_codes[idx],
            'im_a': im_a.astype(np.float32),
            'im_b': im_b.astype(np.float32),
            'im_a_full': frame_a_cropped.transpose(2,0,1).astype(np.float32),
            'im_b_full': frame_b_cropped.transpose(2,0,1).astype(np.float32),
            'camera_enc_ref': camera_enc_ref,
            'camera_enc_a': camera_enc_a,
            'camera_enc_b': camera_enc_b,
            'tform_ref': tform_ref,
            'tform_a_relative': tform_a_relative,
            'tform_b_relative': tform_b_relative,
            'tform_ref': tform_ref,
            'tform_a': tform_a,
            'tform_b': tform_b,
            'focal_a': focal_y_a,
            'focal_b': focal_y_b,
        }
        return out_dict

class UmiDatasetFromFolder(data.Dataset):
    def __init__(self,root_dirs, focal_lengths):
        super().__init__()
        assert len(root_dirs) == len(focal_lengths)
        
        self.poses = {}
        for root_idx,root_dir in enumerate(root_dirs):
            focal_length = focal_lengths[root_idx]
            
            all_poses = umi_utils.umi_read_poses_from_folder(root_dir, focal_length)
            self.poses.update(all_poses)
            
        print("valid_num_data: ", len(self.poses))
        print("focal_lengths: ", focal_lengths)
        self.sequence_codes = list(self.poses.keys())
        self.im_size = 128
    
    def __len__(self):
        return len(self.sequence_codes)
    
    def sample_pair(self, idx):

        folder_path = self.sequence_codes[idx]
        poses = self.poses[folder_path]
        available_idxs = list(range(len(poses['imgs'])))
        sample_distance, sample_range = poses['sample_param']
        has_gripper_state = poses['has_gripper_state']

        while True:
            id_a = np.random.choice(available_idxs,1,replace=False)[0]
            if has_gripper_state:
                a_gripper_state = poses['gripper_state'][id_a]
                if a_gripper_state < 0:
                    continue
            other_indices = []
            lower_ub = id_a-sample_distance
            if lower_ub >= 0:
                other_indices += list(range(max(0, lower_ub-sample_range), lower_ub+1))
            upper_lb = id_a+sample_distance
            if upper_lb <= len(poses['imgs'])-1:
                other_indices += list(range(upper_lb, min(len(poses['imgs']), upper_lb+sample_range)))
            if has_gripper_state:
                other_indices = [i for i in other_indices if poses['gripper_state'][i] == a_gripper_state]
                if len(other_indices) < 1:
                    continue
            id_b = np.random.choice(other_indices,1,replace=False)[0]
            break
        return id_a,id_b

    def __getitem__(self,idx):
        
        folder_path = self.sequence_codes[idx]
        poses = self.poses[folder_path]
        id_a,id_b = self.sample_pair(idx)
        
        frame_a_path = os.path.join(folder_path, poses['imgs'][id_a])
        frame_b_path = os.path.join(folder_path, poses['imgs'][id_b])
        frame_a = Image.open(frame_a_path)
        frame_b = Image.open(frame_b_path)
        
        orig_w, orig_h = frame_a.size
        # Images are rectified to this size, so assert here
        assert orig_w == 256 and orig_h == 256

        im_a = np.copy(np.asarray(frame_a)).transpose(2,0,1)/127.5 - 1
        im_b = np.copy(np.asarray(frame_b)).transpose(2,0,1)/127.5 - 1

        # scale image values
        frame_a_cropped = np.copy(np.asarray(frame_a))/127.5 - 1
        frame_b_cropped = np.copy(np.asarray(frame_b))/127.5 - 1

        tform_a = poses['poses'][id_a]
        tform_b = poses['poses'][id_b]
        
        tform_a_inv = np.linalg.inv(tform_a)
        tform_b_inv = np.linalg.inv(tform_b)
        tform_ref = np.eye(4)
        tform_a_relative = np.matmul(tform_b_inv,tform_a)
        tform_b_relative = np.matmul(tform_a_inv,tform_b)
        
        focal_y_a = poses['focal_length'] / float(orig_h)
        focal_y_b = poses['focal_length'] / float(orig_h)

        camera_enc_ref = rel_camera_ray_encoding(tform_ref,self.im_size,focal_y_a)
        camera_enc_a = rel_camera_ray_encoding(tform_a_relative,self.im_size,focal_y_a)
        camera_enc_b = rel_camera_ray_encoding(tform_b_relative,self.im_size,focal_y_b)
        
        out_dict = {
            'sequence_code':self.sequence_codes[idx],
            'im_a': im_a.astype(np.float32),
            'im_b': im_b.astype(np.float32),
            'im_a_full': frame_a_cropped.transpose(2,0,1).astype(np.float32),
            'im_b_full': frame_b_cropped.transpose(2,0,1).astype(np.float32),
            'camera_enc_ref': camera_enc_ref,
            'camera_enc_a': camera_enc_a,
            'camera_enc_b': camera_enc_b,
            'tform_ref': tform_ref,
            'tform_a_relative': tform_a_relative,
            'tform_b_relative': tform_b_relative,
            'tform_ref': tform_ref,
            'tform_a': tform_a,
            'tform_b': tform_b,
            'focal_a': focal_y_a,
            'focal_b': focal_y_b,
        }
        return out_dict

