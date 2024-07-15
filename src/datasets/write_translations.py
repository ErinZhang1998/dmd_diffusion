import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image 

def get_colmap_labels(folder, file='/sparse/0/images.txt', failed_frames = [], focal_length=None, data_type="vime", interval=1):
  """
  Saves relative translations and rotations to labels.json
  :param f: Current folder
  :param file: File to retrieve poses from
  :return:
  """
  file = folder + file
  if not os.path.exists(file):
    print("File " + file + " doesn't exist.")
    return None
  lines = []
  with open(file, 'r') as fi:
    # First few header lines of images.txt
    for i in range(4):
      fi.readline()
    while True:
      line = fi.readline()
      if not line:
        break
      if ("png" in line) or ("jpg" in line):
        lines.append(line)

  lines.sort(key=lambda x: x.split(" ")[-1].strip())

  dct = {}
  dct['poses'] = []
  dct['dependencies'] = [None]
  dct['generation_order'] = []
  dct['imgs'] = []

  all_displacements = []
  for l in range(0, len(lines)-1):
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
        print("Format incorrect! {folder} {cname} {nname}")
        continue
    frame_in_between = end_frame - start_frame
    assert frame_in_between != 0
      
    q1 = np.array([cQX, cQY, cQZ, cQW])
    q2 = np.array([nQX, nQY, nQZ, nQW])

    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)

    t1 = np.array([float(cTX), float(cTY), float(cTZ)]) 
    t2 = np.array([float(nTX), float(nTY), float(nTZ)]) 
    
    t = t1 - rot1.as_matrix() @ (rot2.inv().as_matrix() @ t2)
    trans = (np.linalg.norm(t) / frame_in_between) * interval
    all_displacements.append(trans)
  
  max_displace = np.max(all_displacements)
  if max_displace == 0:
    return None
  dct['scale_factor'] = float(max_displace)
  count = 0
  height = None
  for l in range(len(lines)):
    curr = lines[l]
    cid, cQW, cQX, cQY, cQZ, cTX, cTY, cTZ, _, cname = curr.split()
    if len(failed_frames) > 0 and cname in failed_frames:
      continue
    
    if l == 0:
      height = Image.open(os.path.join(folder, "images", cname)).size[1]
      if data_type == "vime" and height != 1440:
        return None
      if data_type == "apple" and height != 1080:
        return None
    q1 = np.array([cQX, cQY, cQZ, cQW])
    rot1 = R.from_quat(q1)

    t1 = np.array([float(cTX), float(cTY), float(cTZ)])
    t1 = np.array([t1])

    end_row = np.array([0.0, 0.0, 0.0, 1.0])
    end_row = np.array([end_row])

    orb_t1 = orb_to_blender(np.append(np.append(rot1.as_matrix(), t1.T, 1), end_row, 0))
    # dct[cname] = (list(t), r)
    dct['imgs'].append("images/" + cname)
    dct['poses'].append(orb_t1.tolist())
    dct['dependencies'].append([0])
    count = count + 1
    dct['generation_order'].append(count)

  dct['focal_y'] = float(focal_length)/float(height)
  if len(dct['imgs']) == 0:
    return None
  return dct

def orb_to_blender(orb_t):
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
    
    camera_local = np.linalg.inv(orb_t)
    orb_world = np.matmul(camera_local,pre_conversion)
    blender_world = np.matmul(conversion,orb_world)

    return blender_world