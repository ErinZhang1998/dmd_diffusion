import os 
import json
import numpy as np
import collections

def read_pose(file):
    data = json.load(open(file, "r"))
    dct = collections.defaultdict(list)
    frame_names = sorted(list(data.keys()))
    for frame in frame_names:
        pose = data[frame]
        pose = np.asarray(pose)
        orb_pose = orb_to_blender(pose)
        dct['imgs'].append("images/" + frame)
        dct['poses'].append(orb_pose)
        dct['poses_orig'].append(pose)
    return dict(dct)

def orb_to_blender(camera_local):
    pre_conversion = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1],
    ])
    conversion = np.array([
            [1,0,0,0],
            [0,0,1,0],
            [0,-1,0,0],
            [0,0,0,1],
    ])
    
    orb_world = np.matmul(camera_local,pre_conversion)
    blender_world = np.matmul(conversion,orb_world)

    return blender_world