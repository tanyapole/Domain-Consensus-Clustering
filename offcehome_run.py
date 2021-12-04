import torch
import wandb
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset import *  # init_dataset
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
import trainer
import time, datetime
import copy
import numpy as np
import random
import importlib
import os
import argparse

domain_list = {}
domain_list['officehome'] = ['Art','Product','Clipart', 'RealWorld']
domain_list['office'] = ['amazon', 'webcam', 'dslr']

def main(source, target):
    cudnn.enabled = True
    cudnn.benchmark = True

    config, writer = init_config("config/oh.yaml", sys.argv)
    config.source = source
    config.target = target

    Param = importlib.import_module('trainer.{}{}_trainer'.format(config.trainer, config.version))
    if config.setting=='uda':
        config.cls_share = 10
        config.cls_src   = 5
        config.cls_total = 65
    elif config.setting=='osda':
        config.cls_share = 25
        config.cls_src   = 0
        config.cls_total = 65
    elif config.setting=='pda':
        config.cls_share = 25
        config.cls_src   = 40
        config.cls_total = 65


    config.num_classes = config.cls_share + config.cls_src
    config.uk_index=config.cls_share + config.cls_src
    config.transfer_all = 1 
    a,b,c =  config.cls_share, config.cls_src, config.cls_total
    c = c-a-b
    share_classes  = [i for i in range(a)]
    source_classes = [a+i for i in range(b)]
    target_classes = [a+b+i for i in range(c)]
    config.share_classes  = share_classes
    config.source_classes = share_classes + source_classes
    config.target_classes = share_classes + target_classes
    trainer = Param.Trainer(config, writer)
    if config.wandb:
        run = wandb.init(project='original-DCC', 
                name=_get_run_name(config.source, config.target),
                config={'source': config.source, 'target': config.target})
    trainer.train()
    run.finish()

def _get_run_name(source, target):
    return f'original {source} -> {target}'
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--source', choices=domain_list['officehome'], required=True)
    parser.add_argument('--target', choices=domain_list['officehome'], required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    main(args.source, args.target)
