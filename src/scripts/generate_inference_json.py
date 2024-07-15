import os
import sys
import argparse
import pathlib
import json
import numpy as np
from scipy.spatial.transform import Rotation as scir

script_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(os.path.join(src_path, "datasets"))
from write_translations import get_colmap_labels
from slam_utils import read_pose

def main(args):
    if args.suffix is not None:
        output_root = args.output_root + f"_{args.suffix}"
    else:
        output_root = args.output_root
    # create a folder to store all the generated images
    pathlib.Path(output_root+"/images").mkdir(parents=True, exist_ok=True)
    
    data = {}
    for pf, focal_length, img_h in zip(args.data_folders, args.focal_lengths, args.image_heights):
        for folder_name in os.listdir(pf):
            folder_path = os.path.join(pf, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if args.sfm_method == "colmap":
                if os.path.exists(os.path.join(folder_path, "FAIL")):
                    continue
                dct = get_colmap_labels(folder_path, focal_length=focal_length, data_type="apple")
            elif args.sfm_method == "grabber_orbslam":
                json_path = os.path.join(folder_path, "raw_labels.json")
                if not os.path.exists(json_path):
                    print(f"Skipping {folder_path}, no raw_labels.json")
                    continue
                dct = read_pose(json_path)
            dct["focal_y"] = float(focal_length/img_h)
            data[folder_path] = dct
    
    pre_conversion = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    if args.task == "stack":
        x_euler_range = [-10,10]
        y_euler_range = [-10,10]
        z_euler_range = [-10,15]
        magnitude_l, magnitude_u = 0.02,0.04
    elif args.task == "push":
        # NOTE: pushing data uses COLMAP, thus it doesn't recover real-world metric scale
        magnitude_l, magnitude_u = 0.2, 1.0
        args.sample_rotation = False
    elif args.task == "pour":
        x_euler_range = None
        y_euler_range = None
        z_euler_range = [-10,10]
        magnitude_l, magnitude_u = 0.02,0.04
    elif args.task == "hang":
        x_euler_range = None
        y_euler_range = None
        z_euler_range = [-10,10]
        magnitude_l, magnitude_u = 0.02,0.04
    else:
        raise NotImplementedError
    
    generation_data = []
    folders = sorted(list(data.keys()))
    image_index = 0
    for folder in folders:
        folder_data = data[folder]
        num_imgs = len(folder_data['imgs'])
        
        gripper_state_file = os.path.join(folder, "gripper_state.json")
        if os.path.exists(gripper_state_file):
            gripper_state = json.load(open(gripper_state_file, 'r'))
        else:
            gripper_state = None
        
        for frame_idx in range(0, num_imgs, args.every_x_frame):
            full_img_path = os.path.join(folder, folder_data['imgs'][frame_idx])
            
            # Skip frames where the grabber is opening/closing
            if gripper_state is not None:
                img_key = full_img_path.split("/")[-1]
                if img_key in gripper_state:
                    if gripper_state[img_key] < 0:
                        continue
                
            orig_img_copy_path = os.path.join(output_root, 'images', "%09d_o.png" % image_index)
            
            for i in range (args.mult):
                output_img_path = os.path.join(output_root, "images", "%09d.png" % image_index)
                
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                # Sample a random magnitude between 0 and 5
                magnitude = np.random.uniform(magnitude_l,magnitude_u)
                # Create a translation vector with the random direction and magnitude
                translation = direction * magnitude
                transformation = np.eye(4)
                transformation[:3, 3] = translation
                
                if args.sample_rotation:
                    if x_euler_range is not None:
                        x_euler = np.random.uniform(*x_euler_range)
                    else:
                        x_euler = 0
                    if y_euler_range is not None:
                        
                        y_euler = np.random.uniform(*y_euler_range)
                    else:
                        y_euler = 0
                    if z_euler_range is not None:
                        z_euler = np.random.uniform(*z_euler_range)
                    else:
                        z_euler = 0
                    rot_euler = np.array([x_euler, y_euler, z_euler])
                    rot = scir.from_euler('xyz', rot_euler, degrees=True).as_matrix()
                    transformation[:3,:3] = rot                
                transformation = np.matmul(transformation, pre_conversion)
                transformation = np.matmul(np.linalg.inv(pre_conversion),transformation)
                
                this_data = {
                    "img": full_img_path,
                    "orig_img_copy_path" : orig_img_copy_path,
                    "focal_y": folder_data["focal_y"],
                    "output": output_img_path,
                    "magnitude" : magnitude,
                    "transformation": transformation.tolist()
                }
                if args.sample_rotation:
                    this_data["rot_euler"] = rot_euler.astype(float).tolist()
                generation_data.append(this_data)
                image_index += 1
    
    print("Total number of images:", len(generation_data))
    out_file = os.path.join(output_root, 'data.json')
    if os.path.exists(out_file):
        print(f"{out_file} file exists")
    else:
        with open(out_file, 'w') as f: json.dump(generation_data, f)
        print("Written to: ", out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task',default='push',choices=['push','stack','pour', 'hang'],type=str)
    parser.add_argument('--sfm_method', type=str, choices=['colmap', 'grabber_orbslam'])
    parser.add_argument("--data_folders", type=str, nargs="+", default=[])
    parser.add_argument("--focal_lengths", type=float, nargs="+", default=[])
    parser.add_argument("--image_heights", type=float, nargs="+", default=[])
    parser.add_argument('--output_root', type=str, help="output folder")
    parser.add_argument('--suffix', default=None, help="suffix to add to output_root, if you want to generate multiple versions")
    parser.add_argument('--sample_rotation',action='store_true')
    parser.add_argument('--every_x_frame', type=int, default=10, help="Generate augmenting images of every every_x_frame-th frame. If the demonstrations are recorded at high frame-rate (e.g. above 5fps), nearby frames are very similar, and there are too many frames in each trajectory, so it is not necessary to generate augmenting samples for every frame.")
    parser.add_argument('--mult',type=int,default=3,help='Number of augmenting images to generate per input frame')

    args = parser.parse_args()

    main(args)