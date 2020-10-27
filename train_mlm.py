import argparse
import os
import sys
import logging

import torch
import numpy as np
import random
from trainer_captioning import ocl_train as ocl_train_captioning

from utils.utils import set_config_attr, get_config_attr, CheckpointerFromCfg, mkdir, save_config
from yacs.config import CfgNode

from utils.utils import mkdir, save_config

from nets.mlmcaption import VLBERTModel, LXMERTModel

from ocl import NaiveWrapper, ExperienceReplay, AGEM, ExperienceReplayBalanced

# from apex import amp
from pytorch_transformers import AdamW
from utils.utils import set_cfg_from_args, seed_everything

logger = logging.getLogger(__name__)


def train(cfg):
    algo = get_config_attr(cfg, 'EXTERNAL.OCL.ALGO')
    model_name = get_config_attr(cfg, 'MLMCAPTION.BASE')

    if model_name == 'vlbert':
        base_model = VLBERTModel(cfg, init=True)
    elif model_name == 'lxmert':
        base_model = LXMERTModel(cfg, init=True)
    else:
        raise NotImplementedError('Unknown model type {}'.format(model_name))

    device = 'cuda'
    base_model.to(device)

    optimizer = AdamW(filter(lambda x: x.requires_grad, base_model.parameters()),
                              lr=cfg.SOLVER.BASE_LR, correct_bias=False, eps=1e-4)

    if algo == 'ER':
        model = ExperienceReplay(base_model, optimizer, base_model.cfg)
    elif algo == 'ERB':
        model = ExperienceReplayBalanced(base_model, optimizer, base_model.cfg)
    elif algo == 'AGEM':
        model = AGEM(base_model, optimizer, base_model.cfg)
    elif algo == 'naive':
        model = NaiveWrapper(base_model, optimizer, base_model.cfg)
    else:
        raise ValueError()
    model.to(device)

    arguments = {"iteration": 0, "global_step": 0, "epoch": 0}

    output_dir = cfg.OUTPUT_DIR

    checkpointer = CheckpointerFromCfg(
        cfg, model, optimizer, None, output_dir, save_to_disk=True
    )

    writer = None

    epoch_num = 1 if cfg.EXTERNAL.OCL.ACTIVATED else cfg.EXTERNAL.EPOCH_NUM

    for e in range(epoch_num):
        print("epoch")
        arguments['iteration'] = 0
        epoch = arguments['epoch']
        loss = ocl_train_captioning(model, optimizer,checkpointer,device,arguments,writer,e)
        arguments['epoch'] += 1
        checkpointer.save('model_{:02d}'.format(epoch), **arguments)
        if args.dump_reservoir:
            model.dump_reservoir(os.path.join(cfg.OUTPUT_DIR, 'mem_dump.pkl'))
    return model


def main(args):
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.config)

    cfg.EXTERNAL.EXPERIMENT_NAME = args.name
    cfg.SEED = args.seed
    cfg.DEBUG = args.debug

    cfg.MODE = 'train'
    set_cfg_from_args(args, cfg)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR,
                                  '{}_{}'.format(cfg.EXTERNAL.EXPERIMENT_NAME, cfg.SEED))

    output_dir = cfg.OUTPUT_DIR

    if output_dir:
        mkdir(output_dir)

    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    model = train(cfg)

    output_args_path = os.path.join(output_dir, 'args.txt')
    wf = open(output_args_path, 'w')
    wf.write(' '.join(sys.argv))
    wf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='%id')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dump_reservoir', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)
