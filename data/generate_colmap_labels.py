import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
import copy
import argparse

def get_colmap_labels(folder, file='sparse/0/images.txt', interval = 1, data_type = "apple"):
  image_txt_file = os.path.join(folder, file)
  if not os.path.exists(image_txt_file):
    if not os.path.exists(image_txt_file):
        print(f"ERROR!!!! File {image_txt_file} doesn't exist.")
        return
  
  # Jan-16 data are collected at a slightly higher frame rate
  if "Jan-16-2024" in folder and interval > 1:
    real_interval = int(np.ceil(interval * 1.5))
  else:
    real_interval = interval

  lines = []
  with open(image_txt_file, 'r') as fi:
    # First few header lines of images.txt
    for i in range(4):
      fi.readline()
    while True:
      line = fi.readline()

      if not line:
        break
      if ("png" in line) or ("jpg" in line):
        # print(line.split(" ")[-1].strip())
        lines.append(line)

  lines.sort(key=lambda x: x.split(" ")[-1].strip())
  num_lines = len(lines)

  # Sometimes COLMAP output is wrong by flipping the left and right side of the world
  # Only way to detect is to check through the trajectories manually.
  # We add a dummy folder named "FLIP" under the trajectory folder to indicate this.
  flip_traj = os.path.exists(os.path.join(folder, "FLIP"))
  if flip_traj:
    print("Flip: ", folder)
  
  trans_dict = {}
  T_acc = RigidTransform(rotation=np.eye(3),
                                  translation=np.zeros(3),
                                  from_frame="cam_0",
                                  to_frame=f"cam_0")
  trans_dict_next = {}
  for l in range(0, num_lines-1):
    curr = lines[l]
    next_line = lines[l+1]
    cid, cQW, cQX, cQY, cQZ, cTX, cTY, cTZ, _, cname = curr.split()
    nid, nQW, nQX, nQY, nQZ, nTX, nTY, nTZ, _, nname = next_line.split()

    if data_type == "vime":
      try:
        start_frame = int(cname.split(".")[0].split("frame")[-1])
        end_frame = int(nname.split(".")[0].split("frame")[-1])
      except:
        print("Format incorrect! {folder} {cname} {nname}")
        continue
    elif data_type == "apple":
      try:
        start_frame = int(cname.split(".")[0])
        end_frame = int(nname.split(".")[0])
      except:
        print("Format incorrect! Folder={folder} current frame={cname} next frame={nname}! Should be something like xxxxxx.jpg")
        continue
    
    if l == 0:
      trans_dict[0] = (start_frame, cname, copy.deepcopy(T_acc))

    q1 = np.array([cQX, cQY, cQZ, cQW])
    q2 = np.array([nQX, nQY, nQZ, nQW])

    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)

    t1 = np.array([float(cTX), float(cTY), float(cTZ)]) 
    t2 = np.array([float(nTX), float(nTY), float(nTZ)])
    
    T_world_caml = RigidTransform(rotation=rot1.as_matrix(),
                                  translation=t1,
                                  from_frame="world",
                                  to_frame=f"cam_{l}")
    T_world_caml2 = RigidTransform(rotation=rot2.as_matrix(),
                                  translation=t2,
                                  from_frame="world",
                                  to_frame=f"cam_{l+1}")
    T_caml2_caml = T_world_caml * T_world_caml2.inverse()
    if flip_traj:
      T_caml2_caml.translation *= np.asarray([-1,-1,1])
    trans_dict_next[l] = (start_frame, end_frame, cname, nname, T_caml2_caml)
    # for example, (0, 1, 00000.jpg, 00001.jpg, Transformation between the two cameras)  
  
  for l in range(0, num_lines-1):
    start_frame, end_frame, cname, nname, T_caml2_caml = trans_dict_next[l] 
    
    T_caml_cam0 = copy.deepcopy(T_acc * T_caml2_caml)
    T_acc = T_acc * T_caml2_caml
    
    trans_dict[l+1] = (end_frame, nname, T_caml_cam0)
    # (3, 00003.jpg, Transformation from cam_0 to cam_3)
  
  # T_cami_cami+k = T_cam0_cami+k * T_cami_cam0
  # Calculate transformation from cam_i to cam_i+k depending on the interval
  dct = {}
  all_displacements = []
  for l in range(0, num_lines-real_interval):
    framel, frame_namel, T_caml_cam0 = trans_dict[l]
    framelk, frame_namelk, T_camlk_cam0 = trans_dict[l+real_interval]
    
    if framelk - framel > real_interval * 2:
      print(f"Error {folder} {frame_namel} {frame_namelk}")
      continue
    
    img_full_path = os.path.join(folder, "images", cname)
    if not os.path.exists(img_full_path):
      continue
    
    T_camlk_caml = T_caml_cam0.inverse() * T_camlk_cam0
    T_camlk_caml_t = T_camlk_caml.translation.astype(float)
    T_camlk_caml_r = T_camlk_caml.rotation
    T_camlk_caml_r = T_camlk_caml_r.tolist()
    
    dct[frame_namel] = (list(T_camlk_caml_t), T_camlk_caml_r, frame_namelk)
    all_displacements.append(np.linalg.norm(T_camlk_caml_t))
  
  max_displace = np.max(all_displacements)
  
  if interval == 1:
    json_save_path = os.path.join(folder, f'labels.json')
  else:
    json_save_path = os.path.join(folder, f'labels_{interval}.json')
  if os.path.exists(json_save_path):
    os.remove(json_save_path)
  with open(json_save_path, 'w+') as fp:
    json.dump(dct, fp, indent=4, sort_keys=True)
  return dct, max_displace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help="directory to process")
    parser.add_argument('--recursive', action='store_true', help="if true, recursively process all subfolders")
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--txt', default='sparse/0/images.txt')
    args = parser.parse_args()

    if not args.recursive:
      get_colmap_labels(args.dir, args.txt, interval = args.interval)
    else:
      folders = sorted(os.listdir(args.dir))
      for f in folders:
        if not f.startswith(".") and not (".txt" in f) and not (".sh" in f):
          get_colmap_labels(os.path.join(args.dir, f), interval = args.interval)


if __name__ == "__main__":
  main()
