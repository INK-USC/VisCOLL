import logging

import torch
import json
from tqdm import tqdm
from utils.utils import Timer
from dataloader import get_coco_dataloader, get_flickr_dataloader
import os
from utils.utils import get_config_attr

def inference(
        model,
        model_filename,
        device="cuda",
        mute=False,
):
    model.train(False)
    # convert to a torch.device for efficiency
    device = torch.device(device)
    if not mute:
        logger = logging.getLogger(__name__)
    total_timer = Timer()
    total_timer.tic()
    torch.cuda.empty_cache()
    cfg = model.cfg
    dataset_name = get_config_attr(model.cfg, 'MLMCAPTION.DATASET', default='c')
    if dataset_name.startswith('c'):
        data_loader = get_coco_dataloader(cfg, split='val')
        data_loader.dataset.set_task(-1, split='val', novel_comps=cfg.NOVEL_COMPS)
    elif dataset_name.startswith('f'):
        data_loader = get_flickr_dataloader(cfg, split='test')
        data_loader.dataset.set_task(-1, split='test')
    else:
        raise NotImplementedError

    if not mute:
        pbar = tqdm(
            total=len(data_loader),
            desc="Validation in progress"
        )

    postfix = model_filename.replace('.pth','')
    if cfg.NOVEL_COMPS:
        postfix += '_novel_comps'

    gts, preds = [], []
    inp_sents, gt_words, pred_words = [], [], []
    ppl_words, annotation_ids_words, image_ids_words, task_ids_words = [], [], [], []
    tokenizer = data_loader.dataset.vlbert_tokenizer
    acc, total = 0,0
    total_ppl = 0
    with torch.no_grad():
        for iteration, out_dict in enumerate(data_loader):
            # pack features
            captions = out_dict['captions']
            if captions.size(0) == 1: continue
            labels = out_dict.get('labels', None)
            annotation_ids = out_dict.get('annotation_ids', None)
            image_ids = out_dict.get('image_ids', None)
            tasks = out_dict.get('tasks', None)

            bbox_features, bboxes = out_dict['bbox_feats'].flatten(1), \
                                    out_dict['bboxes'].flatten(1)
            feat = torch.cat([bbox_features, bboxes], -1)

            labels = labels.to(device)
            feat = feat.to(device)
            captions = captions.to(device)
            if 'caption_lens' in out_dict:
                caption_lens = out_dict['caption_lens'].to(device)
                ret_dict = model.forward_net(feat, (captions, caption_lens, labels),reduce=False)
            else:
                ret_dict = model.forward_net(feat, captions,reduce=False)

            score = ret_dict['score']
            _, pred = score.max(-1)

            loss = ret_dict['loss'].view(labels.size(0), -1)
            for b in range(labels.size(0)):
                sent = tokenizer.convert_ids_to_tokens(captions[b].cpu().numpy().tolist())
                gt_word_list, pred_word_list, ppl_list = [], [], []
                for t in range(labels.size(1)):
                    if labels[b,t] != -1:
                        gts.append(labels[b,t])
                        preds.append(pred[b,t])
                        gt_word_list.append(tokenizer.convert_ids_to_tokens(labels[b,t].item()))
                        pred_word_list.append(tokenizer.convert_ids_to_tokens(pred[b,t].item()))
                        if labels[b,t] == pred[b,t]:
                            acc += 1
                        total_ppl += loss[b,t].item()
                        ppl_list.append(loss[b,t].item())
                        total += 1
                inp_sents.append(sent)
                gt_words.append(gt_word_list)
                pred_words.append(pred_word_list)
                ppl_words.append(ppl_list)
                if annotation_ids is not None:
                    annotation_ids_words.append(annotation_ids[b].item() if torch.is_tensor(annotation_ids)
                                                else annotation_ids[b])
                    image_ids_words.append(image_ids[b].item() if torch.is_tensor(image_ids) else image_ids[b])
                if tasks is not None:
                    task_ids_words.append(tasks[b])
            pbar.update(1)
    acc = acc / total
    avg_ppl = total_ppl / total
    print(acc)
    verbose_records = []
    for i, (inp_sent, gt_word_list, pred_word_list, ppl_list) in \
            enumerate(zip(inp_sents, gt_words, pred_words, ppl_words)):
        dic ={
            'inp_sent': inp_sent,
            'gt_word_list': gt_word_list,
            'pred_word_list': pred_word_list,
            'ppl_list': ppl_list,
        }
        if annotation_ids_words:
            dic['annotation_id'] = annotation_ids_words[i]
            dic['image_id'] = image_ids_words[i]
        if task_ids_words:
            dic['task'] = task_ids_words[i]
        verbose_records.append(dic)

    wf = open(os.path.join(cfg.OUTPUT_DIR, 'results_%s.json' % postfix), 'w')
    wf_verbose = open(os.path.join(cfg.OUTPUT_DIR, 'results_verbose_%s.json' % postfix), 'w')
    json.dump({
        'acc': acc,
        'avg_ppl': avg_ppl
    }, wf, indent=4)
    json.dump({
        'records': verbose_records,
    }, wf_verbose, indent=4)
    wf.close()
    wf_verbose.close()
    model.train(True)
    return acc, avg_ppl