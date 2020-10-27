import json
import torch
import pickle
from PIL import Image
import pickle, random
from PIL import Image
from torch.utils.data import Dataset
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
from nltk.stem import WordNetLemmatizer
from nltk import RegexpParser
import h5py
from utils.utils import get_config_attr

CONSTRUCT_DATASET = False

MAIN_DATA_STREAM_FILE = 'datasets/coco_buffer/task_buffer_real_split_%s_split_%s_novel_comps_%s_task_partition_%s.pkl'

CAPTION_ANNOTATIONS = './datasets/coco/annotations/captions_%s2017.json'
SEG_ANNOTATIONS = './datasets/coco/annotations/instances_%s2017.json'

# generally, IMG is not required as we use extracted features
IMAGE_DIR = './datasets/coco/%s2017'

FEAT_BASE = './datasets/coco-features'

IDX_TO_PATH_MAP_TRAIN = './datasets/coco/coco_train_map.json'.format(FEAT_BASE)
IDX_TO_PATH_MAP_VAL = './datasets/coco/coco_val_map.json'.format(FEAT_BASE)

TRAIN_FEAT = '{}/faster_rcnn_coco_train.h5'.format(FEAT_BASE)
VAL_FEAT = '{}/faster_rcnn_coco_val.h5'.format(FEAT_BASE)

OBJ2ID_FILE = 'datasets/coco/obj_id_name.json'

# other files for constructing dataset
NOUN_SYNONYMS = 'datasets/coco_nouns.txt'
POS_FILE = 'datasets/coco_pos_%s.json'
PAIR_OCCURANCE_DIR = 'datasets/occurrences/'
NOVEL_COMPS_DIR = 'datasets/novel_comps/'

class HashableSet(set):
    def __hash__(self):
        return id(self)

def inv_dict(dic):
    return {v: k for k, v in dic.items()}

class COCO(Dataset):
    def __init__(self, split, cfg):
        self.cfg = cfg
        if split == 'dev': split = 'val'
        assert split in ['train', 'val']

        self.debug = hasattr(cfg, 'DEBUG') and cfg.DEBUG
        if self.debug:
            split = 'val'

        self.split = split
        self.pos_tags = None
        self.load_image = False

        # If train, randomly split 5000 images as validation set

        if CONSTRUCT_DATASET:
            self.caption_annotations = json.load(open(CAPTION_ANNOTATIONS % split))
            self.instance_annotations = json.load(open(SEG_ANNOTATIONS % split))
            if split == 'train':
                all_train_image_ids = []
                val_k = 5000
                for annotation in self.caption_annotations['annotations']:
                    image_id = annotation['image_id']
                    all_train_image_ids.append(image_id)
                self.train_val_images = random.sample(all_train_image_ids, val_k)
                print('Choosing %d images from %d images as validation set' %
                      (len(self.train_val_images), len(all_train_image_ids)))
            else:
                self.train_val_images = []

            self.heldout_images, self.heldout_pairs_to_images = get_heldout_images()

            # nltk tools
            self.lemmatizer = WordNetLemmatizer()
            grammar = r"""
              CHUNK:
                {<DT>?<JJ|VBG|VBN>*<NN|NNS>+<VB|VBD|VBG|VBN|VBP|VPZ>*}
            """
            self.parser = RegexpParser(grammar)

            """
            fix the mismatched self.noun_to_task, add a real_task_name dict to map old task name to correct task name
            if you use the datastream from the provided pkl file, can ignore this part
            """
            self.o2rtask, self.r2otask = {}, {}
            self.object_id_to_name, self.object_name_to_id = self._load_coco_objects(self.instance_annotations)
            self.all_tasks = list(self.object_name_to_id.keys())


            self.noun_to_task, self.task_to_noun, self.single_noun_to_full_noun = self._load_word_to_task_mapping()

        self.vlbert_tokenizer = nets.EXPERT.models.VLBertTokenizer.from_pretrained(cfg.EXTERNAL.MLM.TOKENIZER)

        self.current_task = None
        self.current_task_buffer = []

        self.bin_size = 1000
        self.train_feat, self.val_feat = h5py.File(TRAIN_FEAT,'r'), h5py.File(VAL_FEAT,'r')

        self.train_hid2image, self.val_hid2image = json.load(open(IDX_TO_PATH_MAP_TRAIN)), json.load(open(IDX_TO_PATH_MAP_VAL))
        self.train_image2hid, self.val_image2hid = inv_dict(self.train_hid2image), inv_dict(self.val_hid2image)

    def imageid2str(self, image_id: int):
        image_id_str = str(image_id)
        image_id_str = '0' * (12 - len(image_id_str)) + image_id_str + '.jpg'
        return image_id_str

    def _get_image_feats(self, image_id: int):
        image_id_str = self.imageid2str(image_id)
        if image_id in self.train_image2hid:
            feat, hid2image, image2hid = self.train_feat, self.train_hid2image, self.train_image2hid
        elif image_id in self.val_image2hid:
            feat, hid2image, image2hid = self.val_feat, self.val_hid2image, self.val_image2hid
        else:
            raise ValueError(image_id_str)

        hid = int(image2hid[image_id])
        ret_dict = {
            'bbox_feats': torch.from_numpy(feat['bbox_features'][hid]).float(), # [100 (max bbox num), 1024],
            'bboxes': torch.from_numpy(feat['bboxes'][hid]).float(), # [100, 4]
            'bbox_num': torch.from_numpy(feat['num_boxes'][hid]).long(), # int
        }
        return ret_dict

    def _load_coco_objects(self, instance_annotations):
        cache_file = OBJ2ID_FILE
        if os.path.isfile(cache_file):
            dic = json.load(open(cache_file))
            return dic['obj_id2name'], dic['obj_name2id']

        object_name_to_id, object_id_to_name = {}, {}
        # build object names from annotations
        for dic in instance_annotations['categories']:
            object_name_to_id[dic['name']] = dic['id']
            object_id_to_name[dic['id']] = dic['name']

        with open(cache_file,'w') as wf:
            json.dump({'obj_id2name': object_id_to_name, 'obj_name2id': object_name_to_id}, wf)

        return object_id_to_name, object_name_to_id

    def _get_image_to_objects(self):
        dic = defaultdict(list)
        for annotation in self.instance_annotations['annotations']:
            image_id = annotation['image_id']
            dic[image_id].append(annotation['category_id'])
        for image_id in dic:
            dic[image_id] = list(set(dic[image_id]))
        return dic

    def set_task(self, task_id_or_name, split='train', novel_comps=False, novel_objects=False):
        """
        -1 means all tasks
        'continuous' mean continuous non-stationary stream
        :param task_id_or_name:
        :param split:
        :return:
        """

        self._init_continuous_stream(split, novel_comps, novel_objects, save=True, debug=self.debug,
                                     seed=get_config_attr(self.cfg, 'DATA_SEED', default=0),
                                     order=get_config_attr(self.cfg, 'DATA_ORDER', default='random'))

        if task_id_or_name in ['all', -1]:
            random.Random(0).shuffle(self.current_task_buffer)
        # filter image ids
        self.current_task_buffer = [d for d in self.current_task_buffer
                                   if d['annotation']['image_id'] in self.train_image2hid or
                                   d['annotation']['image_id'] in self.val_image2hid]
        print('** Buffer details **')
        print('* length: {}\n* task: {}'.format(len(self.current_task_buffer), task_id_or_name))

    def _load_image(self, image_id):
        image_id_str = str(image_id)
        image_id_str = '0' * (12 - len(image_id_str)) + image_id_str
        path = os.path.join(IMAGE_DIR % self.split, image_id_str + '.jpg')
        # Explicitly call convert to make sure the image was read in rgb mode
        image = Image.open(path).convert('RGB').resize((224,224))
        image = F.to_tensor(image)
        return image, image_id_str

    def __len__(self):
        return len(self.current_task_buffer)

    def __getitem__(self, index):
        instance = self.current_task_buffer[index]
        annotation = instance['annotation']
        task = instance['task']

        no_label_positions = []

        init_tokens = self.vlbert_tokenizer.tokenize(annotation['caption'])
        tokens = self.vlbert_tokenizer.add_special_tokens_vl(init_tokens)
        token_tokens, text_len = self.pad_text(tokens)

        text = self.vlbert_tokenizer.convert_tokens_to_ids(token_tokens)
        text = torch.LongTensor(text).view(-1)
        lm_labels = torch.zeros_like(text).fill_(-1)

        # @@@@@@@@@@@@@@ [CLS] and [IMG] tokens ahead of the sentence @@@@@@@@@@@@@@@@@
        offset = 2

        for mask_idx in range(instance['mask'][0] + offset, instance['mask'][1] + 1 + offset):
            if mask_idx - offset not in no_label_positions:
                lm_labels[mask_idx] = text[mask_idx]
                text[mask_idx] = self.vlbert_tokenizer.convert_tokens_to_ids('[MASK]')
        if self.load_image:
            image, image_id_str = self._load_image(annotation['image_id'])
        else:
            image, image_id_str = None, None
        out_dict = {'caption': text, 'labels': lm_labels, 'caption_len': text_len,
                    'image_id': annotation['image_id'], 'annotation_id': annotation['id'],
                    'task': task}
        if image is not None:
            out_dict['image'] = image
        image_out_dict = self._get_image_feats(annotation['image_id'])
        out_dict.update(image_out_dict)
        return out_dict

    def pad_text(self, txt, l=None):
        """
        :param txt: list of text tokens
        """
        l = l or self.cfg.EXTERNAL.MLM.MAX_TXT_SEQ_LEN
        txt = txt[:l]
        return (txt + [self.vlbert_tokenizer.pad_token for i in range(l - len(txt))]), len(txt)

    def _load_word_to_task_mapping(self):
        """
        load nouns -> fine grained nouns dict and its inverse
        :return:
        """
        f = open(NOUN_SYNONYMS,'r')
        noun_to_task, task_to_noun = {}, defaultdict(list) # noun may be multi words
        lines = f.readlines()
        for i, line in enumerate(lines):
            multiwords = line.strip().split(', ')
            real_task_name = multiwords[0]
            for multiword in multiwords:
                nouns = multiword.split()
                lem_noun = tuple([self.lemmatizer.lemmatize(_) for _ in nouns])
                if i + 1 in self.object_id_to_name:
                    noun_to_task[lem_noun] = self.object_id_to_name[i + 1] # +1 because task id starts from 1
                    task_to_noun[self.object_id_to_name[i + 1]].append(lem_noun)
                    old_task_name = self.object_id_to_name[i + 1]
                    self.r2otask[real_task_name] = old_task_name
                    self.o2rtask[old_task_name] = real_task_name
        # a hash table from last word in noun to task
        single_noun_to_full_noun = defaultdict(list)
        for noun in noun_to_task:
            single_noun_to_full_noun[noun[-1]].append(noun)

        return noun_to_task, task_to_noun, single_noun_to_full_noun

    def _generate_masked_instances(self, annotation):
        ret_instances = []

        annotation_id = annotation['id']
        tokens = self.vlbert_tokenizer.tokenize(annotation['caption'].lower().strip())
        lem_tokens = [self.lemmatizer.lemmatize(_) for _ in tokens]
        noun_to_task, task_to_noun = self.noun_to_task, self.task_to_noun
        single_noun_to_full_noun = self.single_noun_to_full_noun

        # load proprocessed pos data
        if self.pos_tags is None:
            self.pos_tags = json.load(open(POS_FILE % self.split))

        # generate_chunks
        chunks_idx, chunks = [], []
        sent_pos_tags = self.pos_tags[str(annotation_id)]['pos']
        assert len(lem_tokens) == len(sent_pos_tags)
        parse_tree = self.parser.parse([_ for _ in enumerate(sent_pos_tags)])
        chunks = parse_tree.subtrees(filter=lambda x: x.label() == 'CHUNK')
        chunks = [_.flatten() for _ in chunks]
        spans = [(chunk[0][0], chunk[-1][0]) for chunk in chunks]

        def find_span(idx):
            for span in spans:
                if span[0] <= idx <= span[1]:
                    return span
            return None

        for token_idx, token in enumerate(lem_tokens):
            if token in single_noun_to_full_noun:
                possible_nouns = single_noun_to_full_noun[token]
                task_noun, task = None, None
                for noun in possible_nouns: # noun: tuple
                    # test if match
                    match = True
                    for j in range(len(noun)):
                        token_j = token_idx - j
                        if token_j < 0 or lem_tokens[token_j] != noun[-1 - j]:
                            match = False
                            break
                    if match:
                        task_noun = noun
                        # dirty fix
                        task = self.o2rtask[noun_to_task[noun]]
                        break
                # after finding the task noun and the task, find the corresponding chunk and mask those words to
                # generate a traning instance
                if task_noun:
                    span = find_span(token_idx)
                    if span is not None:
                        ret_instances.append({
                            'annotation': annotation,
                            'task': task,
                            'mask': span
                        })
        return ret_instances

    def _init_continuous_stream(self, split, novel_comps, novel_objects, seed=0, order='random', task_partition='any_objects', debug=False, save=True):
        # create continuous stream
        cache_file =  MAIN_DATA_STREAM_FILE % (self.split, split, str(novel_comps), task_partition)
        if os.path.isfile(cache_file) and not debug:
            self.current_task_buffer = pickle.load(open(cache_file, 'rb'))
            print('loading from cache')
            return
        partition_by_tasks = defaultdict(list)
        map_by_tasks = defaultdict(list)

        all_objects = sorted(list(self.object_id_to_name.keys()))
        cnt = 0
        cid2annotation = {}
        visited_annotations = []
        print('building initial stream')
        for annotation in self.caption_annotations['annotations']:
            image_id = annotation['image_id']

            # filter out images for validation from train set
            if split == 'train' and self.split == 'train':
                if image_id in self.train_val_images:
                    continue
            elif split == 'val' and self.split == 'train':
                if image_id not in self.train_val_images:
                    continue

            instances = self._generate_masked_instances(annotation)

            # filter out images for testing compositional generalization
            if not novel_comps:
                if image_id in self.heldout_images:
                    cnt += 1
                    continue
            else:
                if image_id not in self.heldout_images:
                    continue
            for instance in instances:
                partition_by_tasks[self.object_name_to_id[instance['task']]].append(instance)
                visited_annotations.append(instance)

        print('finish initial stream')
        if split == 'train' or self.debug:
            stream = self._build_non_stationary_stream(partition_by_tasks, seed, order)
        else:
            stream = visited_annotations

        self.current_task_buffer = stream
        if save:
            flg = True
            if os.path.isfile(cache_file):
                s = input('confirm (Y) to save the stream at %s' % cache_file)
                if s.strip() != 'Y':
                    flg = False
                    print('Will not save')
            if flg:
                f = open(cache_file,'wb')
                pickle.dump(self.current_task_buffer, f)
                f.close()

    def _build_non_stationary_stream(self, partition_by_task, seed, order):
        prev = 0
        cum_start = [0]

        partition_by_task_keys = list(partition_by_task.keys())
        random.Random(seed).shuffle(partition_by_task_keys)

        for k in partition_by_task_keys:
            cum_start.append(prev + len(partition_by_task[k]))
            prev = cum_start[-1]

        # create spread for each object
        bin_size = self.bin_size
        bin_num = ((cum_start[-1] - 1) // bin_size + 1) * 2
        bins = np.zeros((max(partition_by_task.keys()) + 1, bin_num), dtype=int)
        pbar = tqdm(total=len(partition_by_task_keys))
        for i, k in enumerate(partition_by_task_keys):
            #mu = cum_start[i]
            mu = cum_start[i + 1] - len(partition_by_task[k]) * 0.5
            sigma = max(3000, 0.5 * len(partition_by_task[k]))
            # sigma = random.uniform(0.2 * len(partition_by_tasks[k]), 0.8 * len(partition_by_tasks[k]))
            total = len(partition_by_task[k])
            filled = 0
            for b in range(bin_num):
                cum_z = norm.cdf((bin_size * b - mu) / sigma)
                bins[k, b] = int((cum_z * total) - filled)
                filled += bins[k, b]
            pbar.update(1)
        cum_ptr = [0] * bins.shape[0]
        stream = []
        for b in range(bin_num):
            buff = []
            for k in partition_by_task_keys:
                n = bins[k, b]
                buff.extend(partition_by_task[k][cum_ptr[k]: cum_ptr[k] + n])
                cum_ptr[k] += n
            random.Random(seed).shuffle(buff)
            stream.extend(buff)

        return stream

def get_pair_occurrence():
    base_dir = PAIR_OCCURANCE_DIR
    pair_occurrence = defaultdict(list)
    for file_name in os.listdir(base_dir):
        pair_name = file_name[:-5]
        f = open(base_dir + file_name)
        data = json.load(f)
        for image_id in data['adjective_noun_occurrence_data']:
            if data['adjective_noun_occurrence_data'][image_id]['pair_occurrences']:
                pair_occurrence[pair_name].append(image_id)
    return pair_occurrence

def get_heldout_images():
    base_dir = NOVEL_COMPS_DIR
    cache_file = base_dir + 'cache.json'
    if not os.path.isfile(cache_file):
        all_heldout_pairs = []
        for file_name in os.listdir(base_dir):
            f = open(base_dir + file_name)
            data = json.load(f)
            heldout_pairs = data['heldout_pairs']
            all_heldout_pairs.extend(heldout_pairs)

        pair_occurrence = get_pair_occurrence()
        heldout_pairs_to_images = {k: v for (k,v) in pair_occurrence.items() if k in all_heldout_pairs}
        heldout_images = []
        for k,v in heldout_pairs_to_images.items():
            heldout_images.extend(v)
        f = open(cache_file, 'w')
        json.dump({'heldout_images':heldout_images, 'heldout_pairs_to_images': heldout_pairs_to_images}, f)
        f.close()
    else:
        f = open(cache_file)
        dic = json.load(f)
        heldout_images, heldout_pairs_to_images = dic['heldout_images'], dic['heldout_pairs_to_images']
    heldout_images = set([int(x) for x in heldout_images])
    heldout_pairs_to_images = {k: [int(_) for _ in v] for k,v in heldout_pairs_to_images.items()}
    return heldout_images, heldout_pairs_to_images

