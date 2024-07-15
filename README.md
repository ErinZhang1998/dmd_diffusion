# Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning

This repository contains the diffusion model code of DMD, and it is developed based on code from the paper _Long-Term Photometric Consistent Novel View Synthesis with Diffusion Models_ ([Project Page](https://yorkucvil.github.io/Photoconsistent-NVS) [Github Repo](https://github.com/YorkUCVIL/Photoconsistent-NVS)).

## Table of contents

1. [Python environment](#python-environment)
2. [Prepare Data](#prepare-data)
   i. [ORB-SLAM](#orb-slam)
   ii. [COLMAP](#colmap)
3. [Finetuning from Pre-trained Diffusion Model](#finetuning-from-pre-trained-diffusion-model)
4. [Inference](#inference)

## Python environment

1. Install miniforge following guidelines from this [page](https://github.com/conda-forge/miniforge).
2. Install the environment:

```
conda env create -f environment.yaml
```

## Prepare Data

### ORB-SLAM

1. We modified the ORB-SLAM pipeline implementation from [UMI](https://github.com/real-stanford/universal_manipulation_interface). Modified code can be found under in [Modified Code](https://github.com/ErinZhang1998/universal_manipulation_interface.git).

   i. We used a [grabber](https://www.walmart.com/ip/RMS-19-Grabber-Reacher-Rotating-Gripper-Mobility-Aid-Reaching-Assist-Tool-Trash-Picker-Litter-Pick-Up-Garden-Nabber-Arm-Extension-Ideal-Wheelchair-Di/917952605) to collect demonstrations.
   We follow the same procedure as described in [data collection](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Instruction-4db1a1f0f2aa4a2e84d9742720428b4c), but without gripper calibration as there is no continuous gripper control.

   ii. Since the camera is a wrist-mounted GoPro, the grabber is constant across all frames, so we mask out the grabber. The grabber mask we used is in `data/mask_box_maxlens.png`.

   iii. First, set up docker and pull down the docker image (`chicheng/openicc:latest` and `erinzhang1998/orbslam:latest`).

   iv. Example command:

   ```

   cd PATH_TO_MODIFIED_CODE

   python run_slam_pipeline.py --slam_mask_path [PATH of GRABBER MASK] [FOLDER CONTAINING GOPRO COLLECTED VIDEOS]

   ```

2. Run the provided code above. This will generate a file named `dataset_plan.pkl`.
   Execute the `data/generate_slam_labels.py` script. This script will extract frames from the videos and create a file named **raw_labels.json**, which contains the camera poses of the extracted frames.

   For example:

   ```
   cd data

   python generate_slam_labels.py --input_dir [A LIST OF FOLDERS, EACH CONTAINING THE FILE dataset_plan.pkl] \
   --output_dir [OUTPUT FOLDER] \
   --out_fov 69 69 --out_res 360 360 \
   --save_frame --save_absolute_poses
   ```

   The `raw_labels.json` file contains the camera poses. An example is as follows:

   ```
   {
    "00002.jpg": [
       [0.986649811267853, -0.0519668497145176, -0.154342621564865, -0.161981582641602],
       [0.00200064270757139, 0.951518476009369, -0.307585030794144, 0.105451449751854],
       [0.162844091653824, 0.303169935941696, 0.938919484615326, 0.0346916243433952],
       [0, 0, 0, 1]
   ],
   "00003.jpg": [
       [0.986701369285584, -0.0516649074852467, -0.154114067554474, -0.161879867315292],
       [0.0017330339178443, 0.951430082321167, -0.307860016822815, 0.105277061462402],
       [0.162534326314926, 0.303498804569244, 0.938866913318634, 0.0347107946872711],
       [0, 0, 0, 1]
   ],
   ...
   }
   ```

   - Each key is a frame image filename.
   - The first value is a 3D translation vector.
   - The second value is a 3x3 rotation matrix.

3. Train your own grabber state classifier.

   The classifier should label each frame to indicate whether the grabber is opened, closed, opening, or closing. For training purposes, pairs of frames (A, B)—where frame B is generated from frame A—should ideally both have the grabber in the same state (either both opened or both closed). Frames where the grabber is in the process of opening or closing should be excluded from training. This restriction helps to improve the quality of the generated data.

   The **gripper_state.json** file should follow this format:

   In the code, if this file is provided, negative values (-1 and -2) are used to filter out frames where the grabber is in the middle of closing or opening.

   ```

   {
   "00002.jpg": 0,
   "00003.jpg": 0,
   "00004.jpg": 0, \\ 0: opened
   ...
   "00252.jpg": -1,
   "00253.jpg": -1,
   "00254.jpg": -1, \\ -1: closing
   ...
   "00285.jpg": 1, \\ 1: closed
   "00286.jpg": 1,
   "00287.jpg": 1,
   ...
   "01111.jpg": -2, \\ -2: opening
   "01112.jpg": -2,
   }

   ```

4. Folder structure for training the diffusion model.

   The data used to train the diffusion model should be organized in the following folder structure:

   ```

   ├── traj0
   │ ├── images \\ Saved camera stream
   │ │ ├── 00000.jpg
   │ │ ├── 00001.jpg
   │ │ ├── ...
   │ ├── raw_labels.json \\ ORB-SLAM camera pose output
   │ └── gripper_state.json \\ Optinal, grabber state of each frame in images
   ├── traj1
   │ ├── images
   │ │ ├── 00000.jpg
   │ │ ├── ...
   │ └── ...
   ├── ...

   ```

5. Generate relative transformation JSON file.

   This script generates a JSON file containing the relative transformations between frame `t` and frame `t+k`.

   For example, the following command generates a file **labels_10.json**, which contains the relative camera transformations between frames.

   ```
   cd data

   python generate_slam_labels.py --input_dir [A LIST OF FOLDERS, EACH CONTAINING THE FILE dataset_plan.pkl] \
   --output_dir [OUTPUT FOLDER] \
   --save_relative_poses --intervals [k]
   ```

   The **labels_10.json** file follows this format:

   ```
   {
     "00002.jpg": [
     [-0.00167186072561043, 0.000208774320748506, 0.000869603511211403],
     [
     [0.999986635538126, 0.00509722810039509, 0.000899006204277741],
     [-0.00510099576095352, 0.999978025685813, 0.00423877255084756],
     [-0.000877381586346315, -0.00424331306068129, 0.999990586320341]
     ],
     "00012.jpg"
     ],
     "00003.jpg": [
     [-0.00132419468385872, 0.000879239100652684, -0.000108844939058827],
     [
     [0.999991361720216, 0.00402422985508871, -0.00109364620863756],
     [-0.00402330847415588, 0.999991558366435, 0.000846488366480352],
     [0.00109704334864502, -0.000842077898671161, 0.999999043935812]
     ],
     "00013.jpg"
     ],
     ...
   }
   ```

   - Each key is a frame image filename.
   - The first value is a 3D translation vector.
   - The second value is a 3x3 rotation matrix.
   - The third value is the filename of the frame that is k=10 frames (the interval can be set to a different value) apart from the key frame.

### COLMAP

1. You could also use COLMAP. The COLMAP script we used is in `data/colmap.sh`.

2. Folder structure for training the diffusion model.

   The data used to train the diffusion model should be organized in the following folder structure:

   ```
   ├── traj0
   │   ├── images                   \\ Saved camera stream
   │   │   ├── 00000.jpg
   │   │   ├── 00001.jpg
   │   │   ├── ...
   │   ├── sparse                   \\ COLMAP camera pose output
   │   └── db.db                    \\ Part of COLMAP output
   ├── traj1
   │   ├── images
   │   │   ├── 00000.jpg
   │   │   ├── ...
   │   └── ...
   ├── ...
   ```

3. Generate relative transformation JSON file.

   This script generates a JSON file containing the relative transformations between frame `t` and frame `t+k`.

   ```
   cd data

   python generate_colmap_labels.py \
   --dir [PATH TO ONE TRAJ FOLDER]
   --interval [k]
   --txt 'sparse/0/images.txt'
   ```

   The file follows this format:

   ```
   {
       "00006.jpg": [
           [-0.10417322825562, -0.110055124255363, 0.023932504321829],
           [
           [0.999966800033549, 0.00784759645939038, 0.00219409673298111],
           [-0.00783980033779421, 0.999963003942465, -0.00353953062210531],
           [-0.00222179236803007, 0.00352221182949903, 0.999991328793656]
           ],
           "00007.jpg"
       ],
       "00007.jpg": [
           [0.0666075438300733, -0.0282637156232985, 0.00374081325355435],
           [
           [0.999987458227272, 0.00500092106544807, -0.000272353918039759],
           [-0.00500118823026385, 0.999987004652692, -0.000989263373461276],
           [0.000267403150662385, 0.000990613059554244, 0.999999473590522]
           ],
           "00008.jpg"
       ],
       ...
   }
   ```

   - Each key is a frame image filename.
   - The first value is a 3D translation vector.
   - The second value is a 3x3 rotation matrix.
   - The third value is the filename of the frame that is k=1 frame (the interval can be set to a different value) apart from the key frame.

## Finetuning from Pre-trained Diffusion Model

### Training Commands

Inside `src/scripts/mini_train.py`, you will find a list of training parameters. Here’s a breakdown of some key parameters.

**num_gpus_per_node** (default: 4)\
The number of GPUs to use for training.

**instance_data_path**\
Directory to save the training output. This should include two folders: `checkpoints` (for model checkpoints) and `logs` (for TensorBoard logs). Each checkpoint is around 1GB, so ensure the location has sufficient storage.

**batch_size**\
Choose an appropriate batch size based on your data size.

**colmap_data_folders**\
A list of folders containing the training data. Each folder should follow the structure outlined in the [ORBSLAM](#orb-slam) or [COLMAP](#colmap) sections.

**focal_lengths**\
A list of focal lengths for the cameras used to capture the training data.\
:warning: This list must be the same length as the _colmap_data_folders_ list.

**max_epoch** (default: 10000)\
The maximum number of epochs to train the diffusion model.

The following parameters determine which checkpoints to save:

**n_kept_checkpoints**(default: 5)\
The script saves a checkpoint after every epoch to synchronize across multiple GPUs. It keeps the latest `n_kept_checkpoints` checkpoints on disk. For example, if `n_kept_checkpoints` is 5, the 1st epoch's checkpoint will be deleted after the 6th epoch.

**checkpoint_retention_interval**(default: 200)\
Every `checkpoint_retention_interval`-th checkpoint is saved to disk and is not deleted.

### Download Pre-trained Model Weights

1. We finetune our diffusion model from a model pre-trained on large internet-scale data. The pre-trained model weights can be found at [Pre-trained Weights](https://github.com/YorkUCVIL/Photoconsistent-NVS?tab=readme-ov-file#pretrained-weights). Please download both `RealEstate10K VQGAN weights` and `RealEstate10K diffusion model weights`.
2. Place the "RealEstate10K diffusion model weights" in the `[YOUR instance_data_path to save the training output]/checkpoints` directory, where `[YOUR instance_data_path]` is the directory specified for saving the training output.
3. Update the path of `RealEstate10K VQGAN weights` in the `src/models/vqgan/configs/vqgan_32_4.py` file.

### Pushing

To train a diffusion model for the pushing task, use the following example command:

```

cd src

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --master_port=25678 scripts/mini_train.py \
--batch_size 8 \
--task push \
--sfm_method colmap \
--instance_data_path [OUTPUT FOLDER] \
--colmap_data_folders [INPUT DATA] \
--focal_lengths 1157.5 \
--checkpoint_retention_interval 200

```

`[INPUT DATA]` should follow the format described in the [COLMAP](#colmap) section

### Stacking

To train a diffusion model for the stacking task, use the following example command:

```

cd src

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --master_port=25678 scripts/mini_train.py \
--batch_size 50 \
--task stack \
--sfm_method grabber_orbslam \
--instance_data_path [OUTPUT FOLDER] \
--colmap_data_folders [INPUT DATA] \
--focal_lengths 261.901 \
--max_epoch 20000 \
--checkpoint_retention_interval 500

```

`[INPUT DATA]` should follow the format described in the [ORB-SLAM](#orb-slam) section

### Pouring

```

cd src

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --master_port=25678 scripts/mini_train.py \
--batch_size 49 \
--task pour \
--sfm_method grabber_orbslam \
--instance_data_path [OUTPUT FOLDER] \
--colmap_data_folders [INPUT DATA] [INPUT PLAY DATA] \
--focal_lengths 261.901 261.901 \
--max_epoch 20000 \
--checkpoint_retention_interval 500

```

### Hanging a Shirt

```

cd src

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --master_port=25678 scripts/mini_train.py \
--batch_size 39 \
--task hang \
--sfm_method grabber_orbslam \
--instance_data_path [OUTPUT FOLDER] \
--colmap_data_folders [INPUT TASK DATA] [INPUT PLAY DATA] \
--focal_lengths 261.901 261.901 \
--max_epoch 20000 \
--checkpoint_retention_interval 500

```

## Inference

### Pushing

```[python]
# Generate a JSON file with information about the generated images.
# Run the following command
python scripts/generate_inference_json.py \
--task push --sfm_method colmap \
--data_folders [INPUT DATA] \
--focal_lengths 1157.5 --image_heights 1080 \
--suffix ckpt10000 \
--output_root [OUTPUT DIR] \
--every_x_frame 1 --mult 1

# Output:
# Total number of images: N
# Written to:  [JSON FILE]

# Generate images
# Use the generated JSON file to create images by running the following command:
python scripts/sample-imgs-multi.py [JSON FILE] \
--model [CHECKPOINT PATH] \
--batch_size 32 --gpus 0,1,2,3
```

### Stacking, Pouring, Hanging a Shirt

```[python]
# Generate a JSON file with information about the generated images.
# Run the following command
python scripts/generate_inference_json.py \
--task stack --sfm_method grabber_orbslam \
--data_folders [INPUT DATA] \
--focal_lengths 261.901 --image_heights 360 \
--suffix ckpt10000 \
--output_root [OUTPUT DIR] \
--sample_rotation --every_x_frame 10 --mult 1

# Output:
# Total number of images: N
# Written to:  [JSON FILE]

# Generate images
# Use the generated JSON file to create images by running the following command:
python scripts/sample-imgs-multi.py [JSON FILE] \
--model [CHECKPOINT PATH] \
--batch_size 32 --gpus 0,1,2,3
```

### Annotate generated images with action labels

The following script assigns action labels to the synthesized images.

Example command:

```
cd data

python annotate.py --sfm_method grabber_orbslam \
--inf_json [JSON FILE] \
--gt_json labels_30.json --task_data_json labels_10.json \
--output_dir [OUTPUT DIR OF AUGMENTED DATASET] \
--task_data_dir [TASK DATA DIR]
```

- `gt_json`: a JSON file that contains the relative transformations used as the targets for the synthesized images.
- `task_data_json`: a JSON file of the task data that the augmented dataset will be added to.
