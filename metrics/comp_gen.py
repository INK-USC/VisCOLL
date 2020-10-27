from nltk.stem import WordNetLemmatizer
import numpy as np
import os, json
from collections import defaultdict

lemmatizer = WordNetLemmatizer()


def lem(x):
    return lemmatizer.lemmatize(lemmatizer.lemmatize(x, pos='n'), pos='v')


def tokenize_buffer(buffer, dataset):
    for item in buffer:
        item['tokenized_caption'] = dataset.vlbert_tokenizer.tokenize(item['annotation']['caption'])
    return buffer


def get_synonym_table():
    """
    syn_tab: word to all synonyms;
    syn_inv_tab: word to canonical form
    """
    syn_tab, syn_inv_tab = {}, {}
    dirs = ['./datasets/coco_synonyms/adjectives',
            './datasets/coco_synonyms/nouns',
            './datasets/coco_synonyms/verbs']
    for d in dirs:
        for filename in os.listdir(d):
            path = os.path.join(d, filename)
            word = filename.replace('.json', '')
            data = json.load(open(path))
            syn_tab[word] = data
            for w in data:
                syn_inv_tab[w] = word
    return syn_tab, syn_inv_tab


def get_pairs():
    base_dir = './datasets/occurrences'
    l = []
    for filename in os.listdir(base_dir):
        # data = json.load(open(filename))
        pair = filename.replace('.json', '')
        l.append(pair)
    return l


def find_concept_pair(gt_lem, pairs, syn_inv_tab):
    """
    find from predefined concept pairs
    """
    canon_gt_lem = [syn_inv_tab.get(x, None) for x in gt_lem]
    # print(canon_gt_lem)
    pair = None
    pair_idx = None
    for i in range(len(canon_gt_lem)):
        for j in range(i, len(canon_gt_lem)):
            w1, w2 = canon_gt_lem[i], canon_gt_lem[j]
            if w1 is not None and w2 is not None:
                if '_'.join((w1, w2)) in pairs:
                    pair = '_'.join((w1, w2))
                    pair_idx = i, j
                # w2, w1 = w1, w2
                if '_'.join((w2, w1)) in pairs:
                    pair = '_'.join((w2, w1))
                    pair_idx = j, i
    return pair, pair_idx


def get_possible_seen_pairs_from_novel_pairs(novel_pairs, coco_buffer):
    atom1s = set([x.split('_')[0] for x in novel_pairs])
    atom2s = set([x.split('_')[1] for x in novel_pairs])

    possible_pairs = get_pairs()

    pairs = []
    for x in atom1s:
        for y in atom2s:
            pairs.append((x, y))
    seen = [False] * len(pairs)

    seens = set()

    for item in coco_buffer:
        mask = item['mask']
        masked_words = item['tokenized_caption'][mask[0]: mask[1] + 1]
        lem_masked_words = [lem(_) for _ in masked_words]
        for a1 in atom1s:
            if a1 in lem_masked_words:
                for a2 in atom2s:
                    if a2 in lem_masked_words:
                        cand = '_'.join((a1, a2))
                        if cand in possible_pairs:
                            seens.add(cand)
    return seens  # tuple list


def compute_pair_performance(output_dir, file, novel_pairs, possible_pairs):
    """
    return ppl of each pair {pair: list}
    """
    f = open(os.path.join(output_dir, file))
    dic = json.load(f)['records']
    syn_tab, syn_inv_tab = get_synonym_table()

    # dict: pair -> list of ppls. Only compute ppl on words in a span that are synonyms of the words in the concept pair
    res = defaultdict(list)

    res_by_word_seen = defaultdict(list)
    res_by_word_novel = defaultdict(list)

    for entry in dic:
        gt = entry['gt_word_list']
        gt_lem = [lem(x) for x in gt]
        pair, pair_idx = find_concept_pair(gt_lem, possible_pairs, syn_inv_tab)
        if pair is not None:
            pair_ppl = entry['ppl_list'][pair_idx[0]], entry['ppl_list'][pair_idx[1]]
            res[pair].append(pair_ppl)
            pair_words = pair.split('_')
            if pair in novel_pairs:
                res_by_word_novel[pair_words[0]].append(pair_ppl[0])
                res_by_word_novel[pair_words[1]].append(pair_ppl[1])
            else:
                res_by_word_seen[pair_words[0]].append(pair_ppl[0])
                res_by_word_seen[pair_words[1]].append(pair_ppl[1])
    return res, res_by_word_seen, res_by_word_novel

def compute_novel_seen_performance_verbose(output_dir, file, novel_pairs, possible_pairs):
    """

    :param output_dir:
    :param file:
    :param novel_pairs:
    :param possible_pairs:
    :return:
    """
    pair_performance, ppl_word_seen, ppl_word_novel = compute_pair_performance(output_dir, file, novel_pairs,
                                                                               possible_pairs)

    novel_performance_by_pair = {k: v for k, v in pair_performance.items() if k in novel_pairs}
    seen_performance_by_pair = {k: v for k, v in pair_performance.items() if k in possible_pairs and k not in novel_pairs}
    novel_performance_list, seen_performance_list = [], []
    for k, v in novel_performance_by_pair.items():
        novel_performance_list.extend(v)
    for k, v in seen_performance_by_pair.items():
        seen_performance_list.extend(v)

    novel, seen = np.mean(novel_performance_list), np.mean(seen_performance_list)

    avg_ppl_word_seen, avg_ppl_word_novel = {k: np.mean(v) for k, v in ppl_word_seen.items()}, \
                                            {k: np.mean(v) for k, v in ppl_word_novel.items()}

    return novel, seen, novel_performance_list, seen_performance_list, avg_ppl_word_novel, avg_ppl_word_seen


novel_pairs_v = ["eat_man", "lie_woman",
                 "ride_woman", "fly_bird",
                 "hold_child", "stand_bird",
                 "eat_horse", "stand_child"]
novel_pairs_a = [
    "black_cat", "big_bird", "red_bus", "small_plane", "brown_dog", "small_cat",
    "white_truck", "big_plane", "white_horse", "big_cat", "blue_bus", "small_table",
    "black_bird", "small_dog", "white_boat", "big_truck",
]

novel_pairs = novel_pairs_v + novel_pairs_a