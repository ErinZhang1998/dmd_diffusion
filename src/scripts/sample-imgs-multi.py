import os
import pathlib
import sys

script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunks_num(lst, n):
    """Yield n evenly sized chunks from lst."""
    low = len(lst)//n
    rem = len(lst)-(low*n)
    counts = [low]*n
    for i in range(rem): counts[i] += 1
    ptr = 0
    res = []
    for count in counts:
        res.append(lst[ptr:ptr+count])
        ptr += count
    return res

def gpu_list(string):
    return [int(item) for item in string.split(',')]

import argparse
argParser = argparse.ArgumentParser(description="start/resume inference on a given json file")
argParser.add_argument("config")
argParser.add_argument("-g","--gpus",dest="gpus",action="store",default=[0],type=gpu_list, help="Comma-separated list of GPU IDs to use")
argParser.add_argument("-s",dest="steps",action="store",default=500,type=int,help="Number of steps of diffusion to run for each sample")
argParser.add_argument("-b","--batch_size",dest="batch_size",action="store",default=16,type=int,help="Batch size for sampling")
argParser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
argParser.add_argument("--model", help="Path to model checkpoint",required=True)
cli_args = argParser.parse_args()
assert cli_args.model is not None

from utils import *

from models.score_sde import ncsnpp
import models.score_sde.sde_lib as sde_lib
from models.score_sde.ncsnpp_dual import NCSNpp_dual
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
import json
from models.score_sde.configs.LDM import LDM_config
import shutil
from models.vqgan.vqgan import VQModel
from models.vqgan.configs.vqgan_32_4 import vqgan_32_4_config
from models.score_sde.ema import ExponentialMovingAverage
import functools
from models.score_sde.layerspp import ResnetBlockBigGANpp
import torch.nn.functional as F
import pudb
from filelock import FileLock
import time
from cv2 import cv2
# center crop image to shape
def center_crop(img,shape):
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

def crop_like_dataset(frame_a):
	orig_w, orig_h = frame_a.size
	if orig_w == 256 and orig_h == 256:
		return np.asarray(frame_a)
	# ensure images have 360 height
	if orig_h != 360:
		new_w = round(frame_a.size[0] * (360/frame_a.size[1]))
		frame_a = np.asarray(frame_a.resize((new_w,360)))
	else:
		frame_a = np.asarray(frame_a)
	if orig_w != 360: # if the width is not 360
		# crop and downsample
		left_pos = (frame_a.shape[1]-360)//2
		frame_a_cropped = frame_a[:,left_pos:left_pos+360,:]
	else:
		frame_a_cropped = frame_a
	im_a = Image.fromarray(frame_a_cropped).resize((256,256))
	return np.asarray(im_a)

def inference(data,device):
	# vqgan
	vqgan = VQModel(**vqgan_32_4_config).to(device)
	vqgan.eval()

	# ========================= build model =========================
	config = LDM_config()
	score_model = NCSNpp_dual(config)
	score_model.to(device)
	sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=cli_args.steps)

	# ray downsampler
	ResnetBlock = functools.partial(ResnetBlockBigGANpp,
									act=torch.nn.SiLU(),
									dropout=False,
									fir=True,
									fir_kernel=[1,3,3,1],
									init_scale=0,
									skip_rescale=True,
									temb_dim=None)
	ray_downsampler = torch.nn.Sequential(
		ResnetBlock(in_ch=56,out_ch=128,down=True),
		ResnetBlock(in_ch=128,out_ch=128,down=True)).to(device)
	score_sde = Score_sde_model(score_model,sde,ray_downsampler,rays_require_downsample=False,rays_as_list=True)

	checkpoint = torch.load(cli_args.model)
	adapted_state = {}
	for k,v in checkpoint['score_sde_model'].items():
		key_parts = k.split('.')
		if key_parts[1] == 'module':
			key_parts.pop(1)
		new_key  = '.'.join(key_parts)
		adapted_state[new_key] = v
	score_sde.load_state_dict(adapted_state)

	ema = ExponentialMovingAverage(score_sde.parameters(),decay=0.999)
	ema.load_state_dict(checkpoint['ema'])
	ema.copy_to(score_sde.parameters())
	# substitute model with score modifier, used to make bulk sampling easier
	modifier = Score_modifier(score_sde.score_model,max_batch_size=cli_args.batch_size)
	score_sde.score_model = modifier

	tlog('Setup complete','note')
	batches = chunks(data, cli_args.batch_size)

	with torch.no_grad():
		for batch in batches:
			if cli_args.verbose:
				for el in batch:
					print(device, el['img'])
			conditioning_ims = []
			ff_refs = []
			ff_as = []
			ff_bs = []
			for el in batch:
				im = Image.open(el['img'])
				im = crop_like_dataset(im)
				
				if not os.path.exists(el['orig_img_copy_path']):
					cv2.imwrite(el['orig_img_copy_path'],cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
				im = im[:,:,:3].astype(np.float32).transpose(2,0,1)/127.5 - 1
				im = torch.Tensor(im).unsqueeze(0).to(device)
				encoded_im = vqgan.encode(im)
				focal_y = el['focal_y']
				tform_ref = np.eye(4)
				tform_a_relative = np.asarray(el['transformation'])
				tform_b_relative = np.linalg.inv(np.asarray(el['transformation']))
				camera_enc_ref = rel_camera_ray_encoding(tform_ref,128,focal_y)
				camera_enc_a = rel_camera_ray_encoding(tform_a_relative,128,focal_y)
				camera_enc_b = rel_camera_ray_encoding(tform_b_relative,128,focal_y)
				camera_enc_ref = torch.Tensor(camera_enc_ref).unsqueeze(0)
				camera_enc_a = torch.Tensor(camera_enc_a).unsqueeze(0)
				camera_enc_b = torch.Tensor(camera_enc_b).unsqueeze(0)
				ff_ref = F.pad(freq_enc(camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
				ff_a = F.pad(freq_enc(camera_enc_a),[0,0,0,0,1,1,0,0])
				ff_b = F.pad(freq_enc(camera_enc_b),[0,0,0,0,1,1,0,0])
				conditioning_ims.append([encoded_im])
				ff_refs.append([ray_downsampler(ff_ref.to(device))])
				ff_as.append([ray_downsampler(ff_a.to(device))])
				ff_bs.append([ray_downsampler(ff_b.to(device))])

			# sampling loop
			sampling_shape = (len(conditioning_ims), 4, 32, 32)
			sampling_eps = 1e-5
			x = score_sde.sde.prior_sampling(sampling_shape).to(device)

			timesteps = torch.linspace(sde.T, sampling_eps, sde.N, device=device)
			for i in tqdm(range(0,sde.N)):
				t = timesteps[i]
				vec_t = torch.ones(sampling_shape[0], device=t.device) * t
				_, std = sde.marginal_prob(x, vec_t)
				x, x_mean = score_sde.reverse_diffusion_predictor(x,conditioning_ims,vec_t,ff_refs,ff_as,ff_bs)
				x, x_mean = score_sde.langevin_corrector(x,conditioning_ims,vec_t,ff_refs,ff_as,ff_bs)

			decoded = vqgan.decode(x_mean)
			intermediate_sample = (decoded/2+0.5)
			intermediate_sample = torch.clip(intermediate_sample.permute(0,2,3,1).cpu()* 255., 0, 255).type(torch.uint8).numpy()
			outputs = [x['output'] for x in batch]
			pathlib.Path(os.path.dirname(outputs[0])).mkdir(parents=True, exist_ok=True)
			for out, im in zip(outputs, intermediate_sample):
				im_out = Image.fromarray(im)
				im_out.save(out)

if __name__ == '__main__':
	import torch.multiprocessing as mp
	data_path = cli_args.config
	import json
	data = json.load(open(data_path))
	print("Total files: ", len(data))
	import os
	data = [item for item in data if not os.path.isfile(item["output"])]
	print("Remaining files: ", len(data))
	devices = [f'cuda:{i}' for i in cli_args.gpus]
	if len(devices) == 1:
		inference(data, devices[0])
	else:
		mp.set_start_method('spawn')
		datas = chunks_num(data,len(devices))
		ar = list(zip(datas,devices))
		with mp.Pool(len(devices)) as p:
			p.starmap(inference, ar)
