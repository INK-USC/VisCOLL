import json
import copy
import torch
import pickle
from PIL import Image
import pickle, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import copy
from collections import defaultdict
import os
from nltk.tokenize import word_tokenize
from nets.EXPERT.masker import Masker
import nets.EXPERT.models
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from collections import namedtuple
from data.coco import COCO
from data.coco import inv_dict
import h5py
import xml.etree.ElementTree as ET
import logging
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

CONSTRUCT_DATASET = False

MAIN_DATA_STREAM_PATH = 'datasets/flickr_buffer/flickr_task_buffer_%s_%d.pkl'

ROOT = './datasets/flickr30k/'
IMAGE_DIR = "./datasets/flickr30k/flickr30k-images"
FEAT_BASE = './datasets/flickr_features'
ANNOTATION_DIR = "./datasets/flickr30k/Annotations"
SENTENCES_DIR = "./datasets/flickr30k/Sentences"

VOCAB_FILE = './datasets/flickr30k/flickr_vocab.txt'
VOCAB_SIZE = 20000
TASK_SEED = 0

IDX_TO_PATH_MAP_TRAIN = 'datasets/flickr30k/flickr_train_map.json'.format(FEAT_BASE)
IDX_TO_PATH_MAP_VAL = 'datasets/flickr30k/flickr_val_map.json'.format(FEAT_BASE)
IDX_TO_PATH_MAP_TEST = 'datasets/flickr30k/flickr_test_map.json'.format(FEAT_BASE)

TRAIN_FEAT = '{}/faster_rcnn_flickr_train.h5'.format(FEAT_BASE)
VAL_FEAT = '{}/faster_rcnn_flickr_val.h5'.format(FEAT_BASE)
TEST_FEAT = '{}/faster_rcnn_flickr_test.h5'.format(FEAT_BASE)

# for constructing data stream
TASK_PKL = 'datasets/flickr_buffer/flickr_tasks.pkl'
POS_FILE =  './datasets/flickr30k/flickr_pos.json'

class HashableSet(set):
    def __hash__(self):
        return id(self)


class Flickr(COCO):
    def __init__(self, split, cfg):
        # no super class init
        self.cfg = cfg
        assert split in ['train', 'val', 'test']
        if cfg.DEBUG:
            split = 'val'
            logging.info('using val instead')
        self.split = split

        logger.info('setting up loader for flickr...\n')
        self.vlbert_tokenizer = nets.EXPERT.models.VLBertTokenizer(vocab_file=VOCAB_FILE, do_basic_tokenize=True,
                                                                   no_wordpiece=True)
        if CONSTRUCT_DATASET:
            self.annotations, self.annotation_list = build_annotation_dic(split)
            self.image2sents, self.image2pos = None, None
            self.load_image = False

            self.current_task = None
            self.current_task_buffer = []

            self.bin_size = 1000
            self.task_num = 1000

            self.lemmatizer = WordNetLemmatizer()

        self.train_feat, self.val_feat, self.test_feat = h5py.File(TRAIN_FEAT,'r'), h5py.File(VAL_FEAT,'r'), \
                                                         h5py.File(TEST_FEAT, 'r')

        self.train_hid2image, self.val_hid2image, self.test_hid2image = json.load(open(IDX_TO_PATH_MAP_TRAIN)), \
                                                                        json.load(open(IDX_TO_PATH_MAP_VAL)), \
                                                                        json.load(open(IDX_TO_PATH_MAP_TEST))
        self.train_image2hid, self.val_image2hid, self.test_image2hid = inv_dict(self.train_hid2image), \
                                                                        inv_dict(self.val_hid2image), \
                                                                        inv_dict(self.test_hid2image)

    def _get_image_feats(self, image_id: int):
        image_id_str = self.imageid2str(image_id)
        if image_id in self.train_image2hid:
            feat, hid2image, image2hid = self.train_feat, self.train_hid2image, self.train_image2hid
        elif image_id in self.val_image2hid:
            feat, hid2image, image2hid = self.val_feat, self.val_hid2image, self.val_image2hid
        elif image_id in self.test_image2hid:
            feat, hid2image, image2hid = self.test_feat, self.test_hid2image, self.test_image2hid
        else:
            raise ValueError(image_id_str)

        hid = int(image2hid[image_id])
        ret_dict = {
            'bbox_feats': torch.from_numpy(feat['bbox_features'][hid]).float(), # [100 (max bbox num), 1024],
            'bboxes': torch.from_numpy(feat['bboxes'][hid]).float(), # [100, 4]
            'bbox_num': torch.from_numpy(feat['num_boxes'][hid]).long(), # int
        }
        return ret_dict

    def _get_image_to_sents(self):
        image2sents = defaultdict(list)
        for image_id, data in self.annotations.items():
            for sdata in data['sentence_data']:
                image2sents[image_id].append(sdata['sentence'])
        return image2sents

    def _image_to_captions_pos(self):
        pos_file = POS_FILE
        pos_data = json.load(open(pos_file))
        return pos_data

    def _get_phrase_pos(self, image_id, si):
        phrases = self.annotations[image_id]['sentence_data'][si]['phrases']
        phrase_offsets = []  # [a, b)
        for phrase_data in phrases:
            phrase_offsets.append(
                (phrase_data['first_word_index'], phrase_data['first_word_index'] + len(phrase_data['phrase'].split())))
        return phrase_offsets

    def stat_nouns_in_captions(self):
        stat = defaultdict(int)
        for image_id in self.image2sents:
            for i in range(len(self.image2sents[image_id])):
                tokens = self.image2sents[image_id][i].lower().split()
                tokens2 = self.image2pos[image_id][i]['tokens']
                pos = self.image2pos[image_id][i]['pos']
                if (len(tokens) != len(pos)):
                    print(tokens, tokens2)
                # extra phrase
                phrase_offsets = self._get_phrase_pos(image_id, i)
                for offset in phrase_offsets:
                    phrase_tokens, phrase_pos = tokens[offset[0]: offset[1]], pos[offset[0]: offset[1]]
                    for w, p in zip(phrase_tokens, phrase_pos):
                        if p.startswith('N'):
                            stat[self.lemmatizer.lemmatize(phrase_tokens[-1])] += 1
        stat = sorted(stat.items(), key=lambda x: -x[1])
        return stat

    def _get_vocab(self):
        if os.path.isfile(VOCAB_FILE):
            dic = json.load(open(VOCAB_FILE))
            id2token = dic['id2token']
            token2id = dic['token2id']
        else:
            assert self.split == 'train'
            freqs = defaultdict(int)
            for instance in self.annotations:
                tokens = word_tokenize(instance['caption'].lower().strip())
                for token in tokens:
                    freqs[token] += 1
            id2token = ['<pad>','<s>','<unk>','</s>']
            word_list = sorted(freqs.keys(), key=lambda x: -freqs[x])
            id2token.extend(word_list[:VOCAB_SIZE - len(id2token)])
            token2id = {v: i for i, v in enumerate(id2token)}
            wf = open(VOCAB_FILE, 'w')
            json.dump({'id2token': id2token, 'token2id': token2id},wf)
            wf.close()
            print('%d word in total', len(word_list))
        return id2token, token2id

    def _get_image_to_types(self):
        dic = defaultdict(list)
        for instance in self.annotations:
            image_id = instance['img_id']
            dic[image_id].extend(instance['obj_ids'])
        for image_id in dic:
            dic[image_id] = list(set(dic[image_id]))
        return dic

    def _mask_phrase(self, tokens, phrase_pos):
        ret = []
        for i, token in enumerate(tokens):
            if phrase_pos[0] <= i < phrase_pos[1]:
                ret.append('[MASK]')
            else:
                ret.append(token)
        return ret

    def set_task(self, task_id_or_name, split='train', save=True, debug=False, novel_comps=False, novel_objects=False):
        """
        -1 means all tasks
        'continuous' mean continuous non-stationary stream
        :param task_id_or_name:
        :param split:
        :return:
        """

        self._init_continuous_stream_flickr(split, save=True, debug=debug)
        if task_id_or_name in ['all', -1]:
            random.Random(0).shuffle(self.current_task_buffer)

    def _load_image(self, image_id):
        image_id_str = str(image_id)
        path = os.path.join(IMAGE_DIR, image_id_str + '.jpg')
        # Explicitly call convert to make sure the image was read in rgb mode
        image = Image.open(path).convert('RGB').resize((224,224))
        image = F.to_tensor(image)
        return image, image_id_str

    def __getitem__(self, index):
        instance = self.current_task_buffer[index]

        phrase_offset = instance['phrase_offset']
        masked_tokens = self._mask_phrase(instance['tokens'], phrase_offset)
        tokens, text_len = self.pad_text(self.vlbert_tokenizer.add_special_tokens_vl(
            instance['tokens']))
        mask_offset = [phrase_offset[0] + 2, phrase_offset[1] + 2]
        if mask_offset[0] >= len(tokens):
            mask_offset[0] = len(tokens) - 1
        if mask_offset[1] >= len(tokens):
            mask_offset[1] = len(tokens) - 1

        masked_tokens, _ = self.pad_text(self.vlbert_tokenizer.add_special_tokens_vl(
            masked_tokens))

        text = self.vlbert_tokenizer.convert_tokens_to_ids(tokens)
        masked_text = self.vlbert_tokenizer.convert_tokens_to_ids(masked_tokens)

        text, masked_text = torch.LongTensor(text), torch.LongTensor(masked_text)
        labels = -torch.ones_like(text)
        for i in range(mask_offset[0], mask_offset[1]):
            labels[i] = text[i]

        masked_text, labels = masked_text.view(-1), labels.view(-1)

        out_dict = {'caption': masked_text , 'labels': labels, 'caption_len': text_len,
                    'image_id': instance['image_id'], 'annotation_id': instance['instance_id']}
        image_out_dict = self._get_image_feats(instance['image_id'])
        out_dict.update(image_out_dict)
        return out_dict

    def _init_continuous_stream_flickr(self, split, save, debug=False, seed=0):
        # create continuous stream if no cache exists

        cache_file = MAIN_DATA_STREAM_PATH % (self.split, seed)

        if os.path.isfile(cache_file) and not debug:
            self.current_task_buffer = pickle.load(open(cache_file, 'rb'))
            return

        self.image2sents = self._get_image_to_sents()
        self.image2pos = self._image_to_captions_pos()

        # build initial stream
        initial_stream = []
        id_to_instance = {}
        id_to_task, task_to_id = [], {}
        partition_by_task = defaultdict(list)
        instance_id = 0

        pbar = tqdm(total=len(self.annotations))
        logger.debug('building initial stream\n')

        # build task name to task id mapping
        task_file = TASK_PKL
        excluded = []
        if self.split == 'train':
            stat_nouns = self.stat_nouns_in_captions()
            tasks = [_[0] for _ in stat_nouns[:self.task_num] if _[0] not in excluded]
            dic = {'stat_nouns': stat_nouns,
                   'tasks': tasks}
            wf = open(task_file, 'wb')
            pickle.dump(dic, wf)
            wf.close()
        else:
            dic = pickle.load(open(task_file, 'rb'))
            stat_nouns = dic['stat_nouns']
            tasks = dic['tasks']

        for task in tasks:
            task_to_id[task] = len(id_to_task)
            id_to_task.append(task)

        for image_id, data in self.annotations.items():
            for i, sdata in enumerate(data['sentence_data']):
                sentence_tokens = sdata['sentence'].lower().split()
                for phrase in sdata['phrases']:
                    phrase_offset = (phrase['first_word_index'], phrase['first_word_index'] + len(phrase['phrase'].split()))
                    sent_pos = self.image2pos[image_id][i]['pos']
                    nouns = [sentence_tokens[i] for i in range(phrase_offset[0], phrase_offset[1])
                             if sent_pos[i].startswith('N')]
                    task = 'NONE'
                    if nouns:
                        task = self.lemmatizer.lemmatize(nouns[-1])
                    if task in tasks:
                        task_id = task_to_id[task]
                        instance = {
                            'tokens': sentence_tokens,
                            'task_id': task_id,
                            'image_id': image_id,
                            'instance_id': instance_id,
                            'phrase_offset': phrase_offset
                        }
                        initial_stream.append(instance)
                        id_to_instance[instance_id] = instance
                        partition_by_task[task_id].append(instance)
                        instance_id += 1
            pbar.update(1)
        logger.debug('building non-stationary stream\n')
        if split == 'train':
            stream = self._build_non_stationary_stream(partition_by_task, seed)
        else:
            stream = initial_stream

        self.current_task_buffer = stream
        if save:
            f = open(cache_file,'wb')
            pickle.dump(self.current_task_buffer, f)
            f.close()


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to
    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index': index,
                                             'phrase': phrase,
                                             'phrase_id': p_id,
                                             'phrase_type': p_type})

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info

def build_annotation_dic(split):
    image_id_file = open(os.path.join(ROOT, split + '.txt'))
    ret_dic, ret_list = {}, []
    for image_id in image_id_file.readlines():
        image_id = image_id.strip()
        annotation_path = os.path.join(ANNOTATION_DIR, image_id + '.xml')
        sentence_path = os.path.join(SENTENCES_DIR, image_id + '.txt')
        annotations = get_annotations(annotation_path)
        sentence_data = get_sentence_data(sentence_path)
        for caption_data in sentence_data:
            ret_list.append({'image_id': image_id, 'caption_data': caption_data})
        ret_dic[image_id] = {
            'sentence_data': sentence_data,
            'annotation_data': annotations
        }
    return ret_dic, ret_list


if __name__ == '__main__':
    from yacs.config import CfgNode
    from maskrcnn_benchmark.config import cfg as maskrcnn_cfg

    combined_cfg = CfgNode(maskrcnn_cfg, None, new_allowed=True)
    combined_cfg.merge_from_file('configs/mlmcaptioning/er.yaml')
    cfg = combined_cfg
    ds = Flickr('train', cfg, None)
    ds.set_task(-1, debug=True)
    print(1)