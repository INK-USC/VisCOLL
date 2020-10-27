from .common import *

def evaluate_bleu_score_from_records(filename):
    data = json.load(open(filename))
    hyps, refs = {}, {}
    for i, item in enumerate(data['records']):
        hyps[i] = [' '.join(item['pred_word_list'])]
        refs[i] = [' '.join(item['gt_word_list'])]

    ev = BLEUEvaluator(n=4)
    scores = ev.compute_score(refs, hyps, 'every')
    scores = np.array(scores).mean(-1)  # [4]
    return scores


def evaluate_ppl_from_records(filename):
    data = json.load(open(filename))
    s = 0
    n = 0
    for item in data['records']:
        s += sum(item['ppl_list']) / len(item['ppl_list'])
        n += 1
    return s / n


def agg_final_performance(methods, seeds=None):
    if seeds is None:
        logger.info('default seeds are 1,2,3')
        seeds = [
            1, 2, 3
        ]
    valid = 0
    output_arr = np.zeros((len(methods), len(seeds)))
    output_arr_bleu = np.zeros((len(methods), len(seeds), 4))

    for i, method in enumerate(methods):
        for j, seed in enumerate(seeds):
            try:
                fv = os.path.join(method.format(seed).replace('results_model', 'results_verbose_model'))

                # not equal to data['avg_ppl'] - ppl should be first averaged over a phrase and averaged over all instances
                ppl = evaluate_ppl_from_records(fv)
                output_arr[i, j] = ppl
                bleu_scores = evaluate_bleu_score_from_records(fv)
                output_arr_bleu[i, j] = bleu_scores
                valid += 1
            except FileNotFoundError:
                print('File not found for {}, seed {}'.format(method, seed))
                pass

    mean_ppl = output_arr.mean(-1)
    std_ppl = output_arr.std(-1)

    mean_bleu = output_arr_bleu.mean(1)
    std_bleu = output_arr_bleu.std(1)
    return mean_ppl, std_ppl, mean_bleu, std_bleu

if __name__ == '__main__':
    file = 'xxx/result_verbose_xxx.json'
    evaluate_bleu_score_from_records(file)