import os
import sys
from os.path import join as pjoin
import torch.multiprocessing as mp # don't chdir before we load this
import argparse

# add parent dir to path
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)

# args
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-r","--rank",dest="rank",action="store",default=0,type=int)
argParser.add_argument("--num_gpus_per_node",dest="n_gpus_per_node",action="store",default=4,type=int)
argParser.add_argument("--num_nodes",dest="n_nodes",action="store",default=1,type=int)

argParser.add_argument("--instance_data_path",type=str, help="path to save checkpoints and logs")
argParser.add_argument("--batch_size",type=int)
argParser.add_argument("--colmap_data_folders", type=str, nargs="+", default=[])
argParser.add_argument("--focal_lengths", type=float, nargs="+", default=[])
argParser.add_argument('--task', type=str, choices=['push', 'stack', 'pour', 'hang', 'umi'])
argParser.add_argument('--sfm_method', type=str, choices=['colmap', 'grabber_orbslam', 'umi'])
argParser.add_argument("--max_epoch",type=int, default=10000)

argParser.add_argument("--n_kept_checkpoints",type=int, default=5)
argParser.add_argument("--checkpoint_interval",type=int, default=500)
argParser.add_argument("--checkpoint_retention_interval",type=int, default=200)


cli_args = argParser.parse_args()

from utils import *

globals.instance_data_path = cli_args.instance_data_path
globals.ckpt_path = os.path.join(cli_args.instance_data_path, "checkpoints")
globals.batch_size = cli_args.batch_size
globals.colmap_data_folders = cli_args.colmap_data_folders
globals.focal_lengths = cli_args.focal_lengths
globals.task = cli_args.task
globals.sfm_method = cli_args.sfm_method
globals.n_kept_checkpoints = cli_args.n_kept_checkpoints
globals.checkpoint_interval = cli_args.checkpoint_interval
globals.max_epoch = cli_args.max_epoch
globals.checkpoint_retention_interval = cli_args.checkpoint_retention_interval

import torch.distributed as dist
import utils.trainers
import socket
import signal
import psutil

def worker(local_rank,node_rank,n_gpus_per_node,n_nodes):
    # set multiprocessing to fork for dataloader workers (might avoid shared memory issues)
    try:
        mp.set_start_method('spawn') # fork
    except:
        pass

    # initialize distributed training
    rank = node_rank*n_gpus_per_node + local_rank
    world_size = n_nodes*n_gpus_per_node

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    trainer = utils.trainers.Score_sde_trainer(local_rank,node_rank,n_gpus_per_node,n_nodes)

    # load checkpoint if it exists
    ckpt_dir = globals.ckpt_path
    checkpoints = os.listdir(ckpt_dir)
    checkpoints = [x for x in checkpoints if x.endswith('.pth')]
    checkpoints.sort()
    if len(checkpoints) > 0:
        latest_checkpoint = pjoin(ckpt_dir,checkpoints[-1])
        if rank == 0: tlog(f'Resuming from {latest_checkpoint}','note')
        trainer.load_checkpoint(latest_checkpoint)

    # attach signal handler for graceful termination
    def termination_handler(sig_num,frame):
        trainer.termination_requested = True
    signal.signal(signal.SIGTERM,termination_handler)

    # do training
    try:
        trainer.train()
    except Exception as e:
        # print which worker failed
        tlog(f'Exception on {socket.gethostname()}, rank {rank}: {e}','error')
        raise e

if __name__ == '__main__':
    if cli_args.rank < 0:
        node_rank = int(os.environ['SLURM_PROCID'])
    else:
        node_rank = cli_args.rank
    tlog(f'{socket.gethostname()}: {node_rank}')
    if node_rank == 0:
        init_instance_dirs(['logs','checkpoints'])

    def main_exit_handler(sig_num,frame):
        parent = psutil.Process(os.getpid())
        children = parent.children()
        for child in children:
            child.send_signal(sig_num)
        tlog(f'SIGTERM received, waiting to children to finish','note')
    signal.signal(signal.SIGTERM,main_exit_handler)

    n_gpus_per_node = cli_args.n_gpus_per_node
    n_nodes = cli_args.n_nodes
    # worker(0, node_rank, n_gpus_per_node, n_nodes)
    mp.spawn(worker,nprocs=n_gpus_per_node,args=(node_rank,n_gpus_per_node,n_nodes))
