import os.path as path
import pickle

from tqdm import tqdm

from flickr30k_entities_utils import get_sentence_data, get_annotations

PATH_FLICKR = "/home/junyi/data/flickr30k"
SPLIT_MAP = {
    'train': "train.txt",
    'dev': "val.txt",
    'val': "test.txt"
}

dataset = {}
types = {}
for split_name in ["train", "dev", "val"]:
    split = []
    file_img_index = open(path.join(PATH_FLICKR, SPLIT_MAP[split_name]), 'r')
    for raw_img_idx in tqdm(file_img_index):
        img_idx = raw_img_idx.strip()
        annos = get_annotations(path.join(PATH_FLICKR, "Annotations", img_idx + '.xml'))
        for sent in get_sentence_data(path.join(PATH_FLICKR, "Sentences", img_idx + '.txt')):
            involve_objs = []

            for phrase in sent['phrases']:
                for type_name in phrase['phrase_type']:
                    if type_name not in types:
                        types[type_name] = len(types)
                    if types[type_name] not in involve_objs:
                        involve_objs.append(types[type_name])

            instance = {
                "id": split_name + '_' + str(len(split)),
                "img_id": img_idx,
                "caption": sent['sentence'],
                "obj_ids": involve_objs,
            }
            split.append(instance)

    dataset[split_name] = split

pickle.dump(dataset, open(path.join(PATH_FLICKR, "data.pkl"), 'wb'))
pickle.dump(types, open(path.join(PATH_FLICKR, "entity_types.pkl"), 'wb'))