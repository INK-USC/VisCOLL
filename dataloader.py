import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from data.coco import COCO
from data.flickr30k import Flickr

_dataset = {

}

class BatchCollator(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        out_dict = {
            'image_ids': [],
            'images': [],
            'gt_bboxes': [],
            'info': [],
            'attribute_labels': [],
            'object_labels': [],
            'cropped_image': [],
            'raw': []
        }

        for item in batch:
            out_dict['image_ids'].append(item['image_id'])
            out_dict['images'].append(item['image'])
            out_dict['gt_bboxes'].append(item['gt_bboxes'])

            gt_bbox = item['gt_bboxes']
            out_dict['attribute_labels'].append(gt_bbox.extra_fields['attributes'])
            out_dict['object_labels'].append(gt_bbox.extra_fields['labels'])
            out_dict['cropped_image'].append(item['cropped_image'])
            out_dict['raw'].append(item)
        return out_dict


class COCOBatchCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        ret_dict = {'caption_lens': []}
        keys = ['image','caption','labels','image_id','bbox_feats','bboxes','bbox_num','bbox_labels','spatial_feat',
                'caption_len','annotation_id','task']
        for item in batch:
            for key in keys:
                if key in item:
                    batch_key = (key + 's') if not key.endswith('s') else key
                    if batch_key not in ret_dict:
                        ret_dict[batch_key] = []
                    ret_dict[batch_key].append(item[key])

        if len(ret_dict['caption_lens']) == 0:
            # do padding'
            max_len = max([len(_) for _ in ret_dict['captions']])
            for i in range(len(ret_dict['captions'])):
                ret_dict['captions'][i].extend([0] * (max_len - len(ret_dict['captions'][i])))

        if 'images' in ret_dict:
            ret_dict['images'] = [T.F.normalize(_, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) for _ in ret_dict['images']]

        for key in ret_dict:
            if torch.is_tensor(ret_dict[key][0]):
                ret_dict[key] = torch.stack(ret_dict[key])
            elif type(ret_dict[key][0]) is int:
                ret_dict[key] = torch.LongTensor(ret_dict[key])
        return ret_dict


def get_coco_dataloader(cfg, split='train',batch_size=None):
    dataset = COCO(split, cfg)
    collator = COCOBatchCollator()
    return DataLoader(
        dataset,
        batch_size=cfg.EXTERNAL.BATCH_SIZE if batch_size is None else batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=collator
    )


_flickr_dataloaders = {}
def get_flickr_dataloader(cfg, split='train', batch_size=None):
    if split not in _flickr_dataloaders:
        dataset = Flickr(split, cfg)
        collator = COCOBatchCollator()
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.EXTERNAL.BATCH_SIZE if batch_size is None else batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=collator
        )
        _flickr_dataloaders[split] = dataloader
    return _flickr_dataloaders[split]
