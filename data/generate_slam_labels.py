import os
import json
import numpy as np
import pickle
import pathlib
import cv2
import av
from collections import defaultdict
from autolab_core import RigidTransform
from tqdm import tqdm

import numpy as np
import argparse
import concurrent.futures
import multiprocessing

black_list = {
    "demo_EXAMPLE1": [(0,97)],
    "demo_EXAMPLE2": [(0,60)],
}

def parse_fisheye_intrinsics():
    json_data = {
        "final_reproj_error": 0.2916398582648,
        "fps": 59.94005994005994,
        "image_height": 2028,
        "image_width": 2704,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0029788958491257,
            "focal_length": 796.8544625226342,
            "principal_pt_x": 1354.4265245977356,
            "principal_pt_y": 1011.4847310011687,
            "radial_distortion_1": -0.02196117964405394,
            "radial_distortion_2": -0.018959717016668237,
            "radial_distortion_3": 0.001693880829392453,
            "radial_distortion_4": -0.00016807228608000285,
            "skew": 0.0,
        },
        "nr_calib_images": 59,
        "stabelized": False,
    }  
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    f = intr_data['focal_length']
    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']
    
    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    return opencv_intr_dict

def check_black_list(frame, range_list):
    for start,end in range_list:
        if frame >= start and frame <= end:
            return True

class FisheyeRectConverter:
    def __init__(self, K, D, out_size, out_fov):
        out_size = np.array(out_size)
        # vertical fov
        out_f = (out_size[1] / 2) / np.tan(out_fov[1]/180*np.pi/2)
        out_fx = (out_size[0] / 2) / np.tan(out_fov[0]/180*np.pi/2)
        out_K = np.array([
            [out_fx, 0, out_size[0]/2],
            [0, out_f, out_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), out_K, out_size, cv2.CV_16SC2)

        self.map1 = map1
        self.map2 = map2
    
    def inverse_forward(self, img):
        distor_img = cv2.remap(img, 
            self.map2, self.map1,
            interpolation=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT)
        return distor_img
    
    def forward(self, img):
        rect_img = cv2.remap(img, 
            self.map1, self.map2,
            interpolation=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT)
        return rect_img
    
def to_relative_pose_json(cam_pose_slam, start_frame, end_frame, output_folder, interval = 12):
    basename = os.path.basename(output_folder)
    black_list_frames = black_list.get(basename, [])
    dct = {}
    for i, frame_idx in enumerate(range(start_frame, end_frame+1)):
        goal_frame = frame_idx + interval
        if goal_frame >= end_frame:
            break
        pose1 = cam_pose_slam[i]
        pose2 = cam_pose_slam[i+interval]
        t1 = pose1[:3,-1]
        rot1 = pose1[:3,:3]
        t2 = pose2[:3,-1]
        rot2 = pose2[:3,:3]
        T_slam_cam1 = RigidTransform(rotation=rot1,
                                  translation=t1,
                                  from_frame=f"cam_{frame_idx}",
                                  to_frame=f"slam")
        T_slam_cam2 = RigidTransform(rotation=rot2,
                                    translation=t2,
                                    from_frame=f"cam_{goal_frame}",
                                    to_frame=f"slam")
        T_cam1_cam2 = T_slam_cam1.inverse() * T_slam_cam2
        
        T_cam1_cam2_t = T_cam1_cam2.translation.astype(float)
        T_cam1_cam2_r = T_cam1_cam2.rotation
        T_cam1_cam2_r = T_cam1_cam2_r.tolist()
        
        frame_name = f"{frame_idx:05}.jpg"
        frame_namelk = f"{goal_frame:05}.jpg"
        if len(black_list_frames) > 0:
            if check_black_list(frame_idx, black_list_frames):
                continue
        dct[frame_name] = (list(T_cam1_cam2_t), T_cam1_cam2_r, frame_namelk)

    json_save_path = os.path.join(output_folder, f'labels_{interval}.json')
    if os.path.exists(json_save_path):
        os.remove(json_save_path)
    with open(json_save_path, 'w+') as fp:
        json.dump(dct, fp, indent=4, sort_keys=True)

def to_absolute_pose_json(cam_pose_slam, start_frame, end_frame, output_folder):
    basename = os.path.basename(output_folder)
    black_list_frames = black_list.get(basename, []) # Have to manually check if the ORB-SLAM output is correct.
    # One way is to plot time vs. x/y/z of all the extracted trajectories, usually the outliers are incorrect camera poses.
    dct = {}
    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        pose1 = cam_pose_slam[i]
        frame_name = f"{frame_idx:05}.jpg"
        if len(black_list_frames) > 0:
            if check_black_list(frame_idx, black_list_frames):
                continue
        dct[frame_name] = pose1.tolist()

    json_save_path = os.path.join(output_folder, f'raw_labels.json')
    if os.path.exists(json_save_path):
        os.remove(json_save_path)
    print(f"Save to {json_save_path}")
    with open(json_save_path, 'w+') as fp:
        json.dump(dct, fp, indent=4, sort_keys=True)
  
def main(args):
    num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    # intr_path = pathlib.Path("/data11/zhang401/umi_redwood/example/calibration/gopro_intrinsics_2_7k.json")
    opencv_intr_dict = parse_fisheye_intrinsics()
    out_res = (args.out_res[0],args.out_res[1])
    out_fov = (args.out_fov[0],args.out_fov[1])

    ipath = pathlib.Path(args.input_dir)
    demos_path = ipath.joinpath('demos')
    plan_path = ipath.joinpath('dataset_plan.pkl')
    plan = pickle.load(plan_path.open('rb'))
    
    fisheye_converter = FisheyeRectConverter(
        opencv_intr_dict['K'],
        opencv_intr_dict['D'],
        out_size=out_res,
        out_fov=out_fov
    )
    
    def save_video_frames(mp4_path, output_image_dir, frame_start, frame_end):
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            for frame_idx, frame in enumerate(container.decode(in_stream)):
                if frame_idx < frame_start:
                    continue
                elif frame_idx < frame_end:
                    img = frame.to_ndarray(format='rgb24')
                    img = fisheye_converter.forward(img)
                    img_path = os.path.join(output_image_dir, f"{frame_idx:05}.jpg")
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
    
    with tqdm(total=len(plan)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for plan_episode in plan:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))
                    
                grippers = plan_episode['grippers']
                n_grippers = len(grippers)
                cameras = plan_episode['cameras']
                n_cameras = len(cameras)
                assert n_grippers == 1, "Only support one gripper implemented"
                assert n_cameras == 1, "Only support one camera implemented"
                
                camera = cameras[0]
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                if not os.path.exists(video_path):
                    print(f"Video {video_path} does not exist")
                    continue
                
                video_start, video_end = camera['video_start_end']
                mp4_path = str(video_path)
                video_name = mp4_path.split('/')[-2]
                if len(args.process_these_folders) > 0 and video_name not in args.process_these_folders:
                    continue
                output_image_dir = os.path.join(args.output_dir, video_name, "images")
                os.makedirs(output_image_dir, exist_ok=True)
                if not os.path.exists(output_image_dir) or args.save_frame:
                    futures.add(executor.submit(save_video_frames, 
                    mp4_path, output_image_dir, video_start, video_end))
                
                cam_pose_slam = grippers[0]["cam_pose_slam"]
                if args.save_absolute_poses:
                    to_absolute_pose_json(cam_pose_slam, video_start, video_end, os.path.join(args.output_dir, video_name))
                if args.save_relative_poses:
                    for interval in args.intervals:
                        to_relative_pose_json(cam_pose_slam, video_start, video_end, os.path.join(args.output_dir, video_name), interval=interval)
                
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='diffusion label')
    parser.add_argument('--output_dir', type=str, help="path to save the task data")
    parser.add_argument('--input_dir', type=str, help="path to the folder containing dataset_plan.pkl")
    parser.add_argument('--intervals', type=int, nargs='+', default=[12], help="intervals to calculate relative poses")
    parser.add_argument('--save_frame',action='store_true')
    parser.add_argument('--save_absolute_poses',action='store_true')
    parser.add_argument('--save_relative_poses',action='store_true')
    parser.add_argument('--out_fov', nargs='+', type=float, default=[69, 69])  
    parser.add_argument('--out_res', nargs='+', type=int, default=[360, 360])  
    parser.add_argument('--process_these_folders', nargs='+', type=str, default=[], help="extract frames/save camera poses for these folders only")
    args = parser.parse_args()

    main(args)