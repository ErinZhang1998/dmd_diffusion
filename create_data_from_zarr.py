import zarr
import numpy as np
from typing import Callable, Dict, Iterable, List, Union, Optional, cast
#import multiprocessing as mp
import math
from functools import partial
import random
import yaml
from PIL import Image
_multiprocessing = None

def get_multiprocessing():
    global _multiprocessing
    if _multiprocessing is None:
        import multiprocessing
        _multiprocessing = multiprocessing
    return _multiprocessing

data_path = '/home/jovyan/datasets/new_zarr_format/pick_red_cube'

class LoadDataset():
    def __init__(self, path):
        self._path = path
        with open(path + "/metadata.json", "r", encoding="utf-8") as f:
                self._task_info_table = yaml.safe_load(f)
                
        episodes = self._task_info_table["tasks"]['task_0']["variations"]['variation_0']["episodes"]
        episode_paths = []
        for episode_info in episodes:
            ep_path = path + "/tasks" + '/task_0' + '/variation_0' +'/' + (episode_info["episode_name"] + ".zarr.zip")
            episode_paths.append([int(episode_info["episode_name"][8:]), ep_path])
            print(episode_paths[-1])
    
        mp = get_multiprocessing()
        cpu_count = mp.cpu_count()
        self._cpu_count = cpu_count
        print('cpu_count:', cpu_count)
        
        with mp.Pool(processes=cpu_count) as pool:
                    results = pool.map(
                        self.extract_trajectory,
                        episode_paths
                    )
                    
    def extract_trajectory(self, episode): #SAMPLE SUCH THAT THE LAST TIMESTEP HAS THE GRIPPER CLOSED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print(episode[0])
        episode_data = zarr.open(
                        store=episode[1],
                        mode="r",
                    )
        if(episode[0] == 512):
           # print(list(episode_data['robot']['manipulator_1']))   
            #print(np.array((episode_data['sensor']['camera']['camera_1']['color']['image_raw'])).shape)
            camera_1_images = np.array((episode_data['sensor']['camera']['camera_1']['color']['image_raw']))
            camera_2_images = np.array((episode_data['sensor']['camera']['camera_2']['color']['image_raw']))
            gripper_states = np.array(episode_data['robot']['gripper_1']['gap']['position'])
            print(list((episode_data['robot']['manipulator_1']['control_tip_pose']))[0])
            print(np.array((episode_data['robot']['manipulator_1']['control_tip_pose'])).shape)
            # print(list(episode_data['robot']['gripper_1']['gap']['position']))
            # print(len(list(episode_data['robot']['gripper_1']['gap']['position'])))
            
        
            
    
    def save_trajectory(self, trajectory):
        im = Image.fromarray(camera_1_images[0])
        abc = 23
        im.save(f"/home/jovyan/nutalapati/test{abc:05}.jpg")
        
# class SaveTrajecory():sode_info["episode_name"][8:]), ep_path])
#     def __init__(self, traj)

a = LoadDataset(data_path)