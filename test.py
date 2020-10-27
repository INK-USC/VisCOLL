import argparse
import os

import torch
import numpy as np
import random
from inference_captioning import inference
from utils.utils import get_exp_id
from yacs.config import CfgNode
from utils.utils import mkdir, save_config
from nets.mlmcaption import MLMModel, VLBERTModel,LXMERTModel
from utils.utils import get_config_attr, set_cfg_from_args, seed_everything, CheckpointerFromCfg
import logging

from ocl import ExperienceReplay, NaiveWrapper, AGEM, ExperienceReplayBalanced

logger = logging.getLogger(__name__)

def test(cfg):
    goal = 'classification'
    if hasattr(cfg, 'CAPTION') or hasattr(cfg, 'MLMCAPTION'): goal = 'captioning'

    is_ocl = hasattr(cfg.EXTERNAL.OCL, 'ALGO') and cfg.EXTERNAL.OCL.ALGO != 'PLAIN'

    model_name = get_config_attr(cfg, 'MLMCAPTION.BASE', default='expert')
    if model_name == 'expert':
        base_model = MLMModel(cfg, init=True)
    elif model_name == 'vlbert':
        base_model = VLBERTModel(cfg, init=True)
    elif model_name == 'lxmert':
        base_model = LXMERTModel(cfg, init=True)

    base_model.cfg = cfg
    device  = 'cuda'
    base_model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, base_model.parameters()),
        lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999)
    )

    algo = cfg.EXTERNAL.OCL.ALGO
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

    if is_ocl:
        try:
            model.load_reservoir(os.path.join(cfg.OUTPUT_DIR, 'mem_dump.pkl'))
        except AttributeError:
            pass

    arguments = {"iteration": 0, "global_step": 0, "epoch": 0}

    output_dir = cfg.OUTPUT_DIR
    checkpointer = CheckpointerFromCfg(
        cfg, model, save_dir=output_dir
    )

    if cfg.LOAD_ITER:
        model_filename = 'model_%s_%s.pth' % (cfg.LOAD_EPOCH, cfg.LOAD_ITER)
    else:
        model_filename = 'model_%s.pth' % cfg.LOAD_EPOCH

    extra_checkpoint_data = checkpointer.load(os.path.join(cfg.OUTPUT_DIR, model_filename),
                                              use_latest=False)
    arguments.update(extra_checkpoint_data)

    inference(model, model_filename)

    # print(obj_f1, attr_f1)
    # else:
    #     acc = few_shot_inference(model, optimizer, checkpointer, cfg)
    #     print(acc)


def main(args):
    if '%id' in args.name:
        exp_name = args.name.replace('%id', get_exp_id())
    else:
        exp_name = args.name

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.config)

    cfg.DEBUG = args.debug
    cfg.EXTERNAL.EXPERIMENT_NAME = exp_name
    cfg.SEED = args.seed

    cfg.LOAD_EPOCH = args.epoch
    cfg.LOAD_ITER = args.iter
    cfg.MODE = 'test'
    cfg.NOVEL_COMPS = args.novel_comps
    set_cfg_from_args(args, cfg)

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR,
                                  '{}_{}'.format(cfg.EXTERNAL.EXPERIMENT_NAME, cfg.SEED))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    test(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--name', type=str, default='%id')
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    parser.add_argument('--epoch', type=str, default='00')
    parser.add_argument('--iter', type=str, default='')
    parser.add_argument('--novel_comps', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
