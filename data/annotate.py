import cv2
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import shutil

from collections import defaultdict

import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from html4vision import Col, imagetable
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation as scir

def geo_dist(R1, R2, eps=1e-7):
    R_diffs = R1 @ R2.T
    # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
    traces = np.trace(R_diffs).sum()
    
    dist = np.arccos(np.clip((traces - 1) / 2, -1 + eps, 1 - eps))
    return dist

def from_blender_frame(transformation):
    pre_conversion = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1],
    ])
    
    transformation = np.asarray(transformation)
    transformation = np.matmul(transformation, np.linalg.inv(pre_conversion))
    transformation = np.matmul(pre_conversion, transformation)

    return transformation

def center_crop(img, shape):
    h,w = shape
    center = img.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    center_crop = img[int(y):int(y+h), int(x):int(x+w)]
    return center_crop

def maximal_crop_to_shape(image,shape,interpolation=cv2.INTER_AREA):
    target_aspect = shape[1]/shape[0]
    input_aspect = image.shape[1]/image.shape[0]
    if input_aspect > target_aspect:
        center_crop_shape = (image.shape[0],int(image.shape[0] * target_aspect))
    else:
        center_crop_shape = (int(image.shape[1] / target_aspect),image.shape[1])
    cropped = center_crop(image,center_crop_shape)
    resized = cv2.resize(cropped, (shape[1],shape[0]),interpolation=interpolation)
    return resized

def main():
    parser = argparse.ArgumentParser(description='diffusion label')
    parser.add_argument('--sfm_method', type=str, choices=['colmap', 'grabber_orbslam'])
    parser.add_argument('--scale_dict', type=str)
    parser.add_argument('--inf_json', type=str, help="Json used by diffusion model to synthesize perturbed views.")
    parser.add_argument('--gt_json', type=str, default="labels.json", help="Json containing the target label for augmenting images")
    parser.add_argument('--task_data_json', type=str, default="labels.json", help="Json file containing labels for real images that are used to train BC policy.")
    
    parser.add_argument('--output_dir', type=str, help="Path to save the augmented dataset.")
    parser.add_argument('--task_data_dir', type=str, help="Path of task data")
    
    parser.add_argument('--min_trans_dist', type=float, default=0.01) 
    parser.add_argument('--min_rot_dist', type=float, default=1.0)
    
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--viz_dir', type=str)
    parser.add_argument("--viz_traj", nargs="+", default=[])

    
    args = parser.parse_args()
    
    if args.sfm_method == "colmap":
        assert args.scale_dict is not None
        scale_dct = json.load(open(args.scale_dict, "r"))
    
    inf_json = json.load(open(args.inf_json, "r"))
    
    augmented_data = defaultdict(list) 
    num_missing = 0
    if args.viz:
        viz_count = 0
        col_name = []
        col_aug_name = []
        col_img = []
        col_prod = []
        col_orig_vec = []
        col_sample_vec = []
        col_anno_vec = []
        col_sample_euler_img = []
        col_anno_euler_img = []
        col_sample_euler = []
        col_anno_euler = []
        col_orig_euler = []
        
        os.makedirs(args.viz_dir, exist_ok=True)
        
    for diff_idx in tqdm(range(len(inf_json))):
        entry = inf_json[diff_idx]
        expert_img_path = entry['img']
        augment_img_path = entry['output']
        src, frame = os.path.split(expert_img_path)
        frame_idx = int(frame.split(".")[0])
        src = os.path.dirname(src) 
        traj_name = os.path.split(src)[1] 
        
        transformation = np.asarray(entry['transformation'].copy())
        
        tTt2 = from_blender_frame(transformation)
        if args.sfm_method == "colmap":
            scale = scale_dct[traj_name]
            tTt2[:3,3] *= scale
        delta_trans = tTt2[:3,3]
        
        expert_labels_path = os.path.join(src, args.gt_json)
        expert_labels = json.load(open(expert_labels_path, "r"))
        frame_augment_name = ("diffusion" + "_" + os.path.split(augment_img_path)[1]) 

        if frame not in expert_labels:
            # print(f"Missing: {src} {frame} in expert labels")
            num_missing += 1
            continue
        
        trans_label, rot_label, frame_k_name = expert_labels[frame]
        trans_label = np.asarray(trans_label)
        rot_label = np.asarray(rot_label)
        
        if np.linalg.norm(trans_label) < args.min_trans_dist and geo_dist(np.eye(3), rot_label) < np.radians(args.min_rot_dist):
            continue
        
        # next frame pose in current frame
        tTts = np.eye(4)
        tTts[:3,:3] = rot_label
        tTts[:3,3] = trans_label
        
        t2Tts = np.matmul(np.linalg.inv(tTt2), tTts)
        anno_trans = t2Tts[:3,3]
        filter_factor = np.dot(trans_label, delta_trans) / (np.linalg.norm(trans_label) ** 2)
        augmented_data[traj_name].append((t2Tts, frame, frame_k_name, filter_factor, frame_augment_name, augment_img_path))
        
        if args.viz:
            
            if len(args.viz_traj) > 0 and traj_name not in args.viz_traj:
                continue
            
            path1 = os.path.join(src, "images", frame)
            path2 = os.path.join(src, "images", frame_k_name)
            i_t = maximal_crop_to_shape(np.asarray(PIL.Image.open(path1)), (224,224), interpolation=cv2.INTER_AREA)
            i_k = maximal_crop_to_shape(np.asarray(PIL.Image.open(path2)), (224,224), interpolation=cv2.INTER_AREA)
            i_td = maximal_crop_to_shape(np.asarray(PIL.Image.open(augment_img_path)), (224,224), interpolation=cv2.INTER_AREA)
            
            # Create an image of It It_delta It+k
            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(20, 10)
            axs[0].imshow(i_t)
            axs[1].imshow(i_td)
            axs[2].imshow(i_k)
            if args.sfm_method == "colmap":
                start_default=112
                rect(axs[0], (trans_label[0], trans_label[1], trans_label[2]), color="green", start_x=start_default, start_y=start_default, width=3, head_width=8, head_length=4, scale_arrow=35)
                endx,endy = rect(axs[1], (delta_trans[0], delta_trans[1], delta_trans[2]), color="purple", start_x=start_default, start_y=start_default, width=3, head_width=8, head_length=4, scale_arrow=35)
                rect(axs[1], (anno_trans[0], anno_trans[1], anno_trans[2]), color="cyan", start_x=endx, start_y=endy, width=3, head_width=8, head_length=4, scale_arrow=35)
                
            else:
                start_default=112
                _ = rect(axs[0], (trans_label[0], trans_label[1], trans_label[2]), color="green", start_x=start_default, start_y=start_default, width=3, head_width=8, head_length=4, scale_arrow=600)
                endx,endy = rect(axs[0], (delta_trans[0], delta_trans[1], delta_trans[2]), color="purple", start_x=start_default, start_y=start_default, width=3, head_width=8, head_length=4, scale_arrow=600)
                endx,endy = rect(axs[1], (delta_trans[0], delta_trans[1], delta_trans[2]), color="purple", start_x=start_default, start_y=start_default, width=3, head_width=8, head_length=4, scale_arrow=600)
                rect(axs[1], (anno_trans[0], anno_trans[1], anno_trans[2]), color="cyan", start_x=endx, start_y=endy, scale_arrow=600)
            fig.savefig(os.path.join(args.viz_dir, f"{viz_count:04d}.png"), dpi=100)
            plt.close(fig)
            
            col_name.append(f"{traj_name} {frame} {frame_k_name}")
            col_aug_name.append(augment_img_path.replace(".png", ""))
            col_img.append(f"{viz_count:04d}.png")
            col_prod.append(f"{filter_factor:.3f}")
            trans_labelcm = trans_label*100
            delta_transcm = delta_trans*100
            transcm = anno_trans*100
            col_orig_vec.append(f"[{trans_labelcm[0]:.2f}, {trans_labelcm[1]:.2f}, {trans_labelcm[2]:.2f}]")
            col_sample_vec.append(f"[{delta_transcm[0]:.2f}, {delta_transcm[1]:.2f}, {delta_transcm[2]:.2f}]")
            col_anno_vec.append(f"[{transcm[0]:.2f}, {transcm[1]:.2f}, {transcm[2]:.2f}]")
            
            if args.sfm_method != "colmap":
                A2B = pt.transform_from(np.eye(3), np.array([0,0,0]))
                B2C = pt.transform_from(t2Tts[:3,:3], t2Tts[:3,3])
                ax = pt.plot_transform(A2B=A2B)
                view = (-90, -90)
                ax.view_init(*view)
                plt.setp(ax, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), zlim=(-0.1,0.1), xlabel="X", ylabel="Y", zlabel="Z")
                pt.plot_transform(ax, A2B=B2C)
                plt.savefig(os.path.join(args.viz_dir, f"{viz_count:04d}_rot.png"), dpi=100)
                plt.close("all")
                
                A2B = pt.transform_from(np.eye(3), np.array([0,0,0]))
                B2C = pt.transform_from(tTt2[:3,:3], tTt2[:3,3])
                ax = pt.plot_transform(A2B=A2B)
                ax.view_init(*view)
                plt.setp(ax, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), zlim=(-0.1,0.1), xlabel="X", ylabel="Y", zlabel="Z")
                pt.plot_transform(ax, A2B=B2C)
                plt.savefig(os.path.join(args.viz_dir, f"{viz_count:04d}_rot_sample.png"), dpi=100)
                plt.close("all")
            
                this_orig_euler = scir.from_matrix(tTts[:3,:3]).as_euler('xyz', degrees=True)
                this_sample_euler = scir.from_matrix(tTt2[:3,:3]).as_euler('xyz', degrees=True)
                this_anno_euler = scir.from_matrix(t2Tts[:3,:3]).as_euler('xyz', degrees=True)
                
                this_orig_geodist = np.degrees(geo_dist(np.eye(3), tTts[:3,:3], eps=1e-7))
                this_sample_geodist = np.degrees(geo_dist(np.eye(3), tTt2[:3,:3], eps=1e-7))
                this_anno_geodist = np.degrees(geo_dist(np.eye(3), t2Tts[:3,:3], eps=1e-7))
                
            
                col_anno_euler_img.append(f"{viz_count:04d}_rot.png")
                col_sample_euler_img.append(f"{viz_count:04d}_rot_sample.png")
                
                col_sample_euler.append(f"[{this_sample_euler[0]:.2f}, {this_sample_euler[1]:.2f}, {this_sample_euler[2]:.2f}] {this_sample_geodist:.2f}")
                col_anno_euler.append(f"[{this_anno_euler[0]:.2f}, {this_anno_euler[1]:.2f}, {this_anno_euler[2]:.2f}] {this_anno_geodist:.2f}")
                col_orig_euler.append(f"[{this_orig_euler[0]:.2f}, {this_orig_euler[1]:.2f}, {this_orig_euler[2]:.2f}] {this_orig_geodist:.2f}")

            viz_count+=1

    if args.viz:
        if args.sfm_method == "colmap":
            cols = [
                Col('text', 'I_t', col_name),
                Col('text', 'Diffusion', col_aug_name),
                Col('text', 'Orig Trans (cm)', col_orig_vec),
                Col('text', 'Sampled Trans (cm)', col_sample_vec), 
                Col('text', 'Anno Trans (cm)', col_anno_vec), 
                Col('text', 'Filter Factor', col_prod),
                Col('img', 'Trans Vis (I_t, I_d, I_t+1)', col_img),
            ]
        else:
            cols = [
                Col('text', 'I_t', col_name),
                Col('text', 'Diffusion', col_aug_name),
                Col('text', 'Orig Trans (cm)', col_orig_vec),
                Col('text', 'Sampled Trans (cm)', col_sample_vec),
                Col('text', 'Anno Trans (cm)', col_anno_vec), 
                Col('text', 'Orig Rot', col_orig_euler),
                Col('text', 'Sampled Rot', col_sample_euler),
                Col('text', 'Anno Rot', col_anno_euler),  
                Col('text', 'Filter Factor', col_prod),
                Col('img', 'Trans Vis (I_t, I_d, I_t+1)', col_img), 
                Col('img', 'Sample Transform Vis', col_sample_euler_img), 
                Col('img', 'Anno Transform Vis', col_anno_euler_img), 
            ]
        imagetable(cols, f'{args.viz_dir}/index.html', sortable=True, 
                sticky_header=True, sort_style='materialize', sortcol=2,
                imscale=0.7)
    else:
        total_length = 0
        total_imgs = 0
        total_orig_lengths = 0
        for traj in sorted(list(augmented_data.keys())):
            traj_output_dir = os.path.join(args.output_dir, traj)
            os.makedirs(traj_output_dir, exist_ok=True)
            orig_lengths, length, num_img = write_augmented_dataset(
                augmented_data[traj],
                traj_output_dir,
                os.path.join(args.task_data_dir, traj), 
                args.task_data_json,
            )
            total_orig_lengths += orig_lengths
            total_length += length 
            total_imgs += num_img
        
        print(f"Original task data have {total_orig_lengths} samples.\nAugmented data include {total_length} samples across {total_imgs} images")

def write_augmented_dataset(traj_augmented_data, traj_output_dir, traj_expert_dir, task_data_json):
    _, traj = os.path.split(traj_output_dir)
    prefix = task_data_json.split(".")[0]
    output_json_path = os.path.join(traj_output_dir, prefix+"_augment.json")
    print("Save to: ", output_json_path)
    length = 0
    num_img = defaultdict(int)
    orig_lengths = 0
    os.makedirs(os.path.join(traj_output_dir, "images"), exist_ok=True)
    with open(output_json_path, "w+") as jsonf:
        label_data = json.load(open(os.path.join(traj_expert_dir, task_data_json), "r"))
        new_output = {}
        for frame,v in label_data.items():
            new_path = os.path.join(traj_output_dir, "images", frame)
            if not os.path.exists(new_path):
                shutil.copy(os.path.join(traj_expert_dir, "images", frame), os.path.join(traj_output_dir, "images", frame))
            new_output[frame] = v
        orig_length = len(new_output)
        orig_lengths += orig_length
        
        for t2Tts, frame, frame_k_name, filter_factor, frame_augment_name, augment_img_path in traj_augmented_data:
            trans,rot = list(t2Tts[:3,3]), t2Tts[:3,:3].tolist()
            num_img[os.path.join(traj,frame)] += 1
            new_output[frame_augment_name] = (trans, rot, filter_factor, frame_k_name)
            
            augment_img_new_path = os.path.join(traj_output_dir, "images", frame_augment_name)
            if not os.path.exists(augment_img_new_path):
                shutil.copy(augment_img_path, augment_img_new_path)

        json.dump(new_output, jsonf, indent=4, sort_keys=True)
        length = len(new_output) - orig_length
        
    return orig_lengths, length, len(dict(num_img))

def rect(ax, poke, min_z=-1, max_z=1, scale_factor=1, pred=False, color="cyan", start_x=150, start_y=150, width=1, head_width=7, head_length=6, scale_arrow=35):
    x, y, z = poke
    # z is positive pointing towards me.
    z = -z
    sign = 1

    dx = scale_arrow * (x / scale_factor)
    dz = scale_arrow * (z / scale_factor)
    
    cmap = cm.seismic
    norm = Normalize(vmin=min_z, vmax=max_z)
    c = cmap(norm(y))
    if norm(y) < 0.5:
        sign = -1

    ax.arrow(start_x, start_y, dx, dz, width=width, head_width=head_width, head_length=head_length, color=color)

    # up down arrow
    if pred:
        ax.arrow(25, 190, 0, sign * 20, width=4, head_width=10, head_length=8, color=c)
    else:
        ax.arrow(10, 190, 0, sign * 20, width=4, head_width=10, head_length=8, color=c)

    return start_x+dx, start_y+dz


if __name__=='__main__':
    main()