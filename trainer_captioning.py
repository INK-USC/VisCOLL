import logging
import torch
from tqdm import tqdm
import json, os
from dataloader import get_coco_dataloader
from collections import defaultdict
from dataloader import get_flickr_dataloader
from utils.utils import get_config_attr


def ocl_train(model,optimizer,checkpointer,device,arguments,writer,epoch):
    logger = logging.getLogger("trainer")
    logger.info("Start training @ epoch {:02d}".format(arguments['epoch']))

    dataset_name = get_config_attr(model.cfg, 'MLMCAPTION.DATASET', default='c')
    if dataset_name.startswith('c'):
        data_loader, val_data_loader = get_coco_dataloader(model.cfg, split='train'), \
                                       get_coco_dataloader(model.cfg, split='val')
    elif dataset_name.startswith('f'):
        data_loader, val_data_loader = get_flickr_dataloader(model.cfg, split='train'), \
                                        get_flickr_dataloader(model.cfg, split='val')
    else:
        raise ValueError('unknown dataset {}'.format(dataset_name))

    start_iter = arguments["iteration"]
    model.train()

    metric_dict = defaultdict(list)

    use_image_feat = get_config_attr(model.cfg, 'EXTERNAL.USE_IMAGE_FEAT', default=0)

    task = 'continuous' if model.cfg.EXTERNAL.OCL.ACTIVATED else 'all'
    dataset = data_loader.dataset

    iteration = 0
    all_train_loss = []

    data_loader.dataset.set_task(task)
    val_data_loader.dataset.set_task(task, split='val')
    pbar = tqdm(
        total=len(data_loader),
    )

    print(task, len(data_loader))

    for iter_num, out_dict in enumerate(data_loader, start_iter):
        pbar.update(1)
        arguments["global_step"] += 1
        arguments["iteration"] = iteration
        # pack features
        captions = out_dict['captions']
        if captions.size(0) == 1: continue
        labels = out_dict.get('labels', None)
        if 'caption_lens' in out_dict:
            caption_lens = out_dict['caption_lens']
        if use_image_feat:
            images = out_dict.get['images']
            if images.size(0) == 1: continue
            feat = images.flatten(1)
        else:
            bbox_features,  bboxes = out_dict['bbox_feats'].flatten(1), \
                                    out_dict['bboxes'].flatten(1)
            feat = torch.cat([bbox_features, bboxes], -1)

        captions = captions.to(device)
        if 'caption_lens' in out_dict:
            assert labels is not None
            feat = feat.to(device)
            caption_lens = caption_lens.to(device)
            labels = labels.to(device)
            captions = captions.to(device)
            ret_dict = model.observe(feat, (captions, caption_lens, labels))
        else:
            ret_dict = model.observe(feat, captions)

        all_train_loss.append(ret_dict['loss'].item())
        iteration += 1

        # switch task
        if (iter_num % 2000 == 0 and iter_num) or iter_num == 1000:
            val_loss = validate(model, val_data_loader, start_iter, use_image_feat, device, metric_dict,
                        all_train_loss, iter_num, epoch, arguments, checkpointer)
    return val_loss

def validate(model, val_data_loader, start_iter, use_image_feat, device, metric_dict,
             all_train_loss, train_iter_num, train_epoch, arguments, checkpointer):
    val_loss = 0
    with torch.no_grad():
        model.eval()
        pbar_val = tqdm(
            total=len(val_data_loader),
        )
        for val_iteration, val_out_dict in enumerate(val_data_loader, start_iter):
            # pack features
            captions = val_out_dict['captions']
            if captions.size(0) == 1:
                continue # prevent layernorm error
            labels = val_out_dict.get('labels', None)

            bbox_features, bboxes = val_out_dict['bbox_feats'].flatten(1), \
                                            val_out_dict['bboxes'].flatten(1)
            feat = torch.cat([bbox_features, bboxes], -1)

            labels = labels.to(device)
            feat = feat.to(device)
            captions = captions.to(device)
            if 'caption_lens' in val_out_dict:
                caption_lens = val_out_dict['caption_lens'].to(device)
                ret_dict = model.forward_net(feat, (captions, caption_lens, labels))
            else:
                ret_dict = model.forward_net(feat, captions)
            loss = ret_dict['loss'].item()
            val_loss += loss
            pbar_val.update(1)
        model.train()
    val_loss /= len(val_data_loader)
    metric_dict['val_loss'].append(val_loss)
    metric_dict['train_loss'] = all_train_loss
    wf = open(os.path.join(model.cfg.OUTPUT_DIR, 'metric_{}_epoch_{}.json'.
                           format(model.cfg.EXTERNAL.BATCH_SIZE * train_iter_num, train_epoch)), 'w')
    json.dump(metric_dict, wf, indent=4)
    wf.close()
    if get_config_attr(model.cfg, 'EXTERNAL.OCL.ACTIVATED', totype=bool):
        checkpointer.save('model_{}_{}'.format(train_epoch, train_iter_num), **arguments)
    return val_loss
