import sys
sys.path.append('../..')
import nltk
from .bleu_scorer import BleuScorer


class BLEUEvaluator(object):
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res, mode='all'):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        if mode == 'all':
            return score
        elif mode == 'every':
            return scores
        else:
            return score, scores

    def method(self):
        return "Bleu"


def main():
    hypo = {'1': ['I like it !'], '2': ['I completely do not know !'],
            '3': ['how about you ?'], '4': ['what is this ?'], 5: ['this is amazing !']}
    ref = {'1': ['I love you !', 'I love myself !', 'I like it !'], '2': ['I do not know !'], '3': ['how are you ?'],
           '4': ['what is this animal ?'], 5: ['this is awkward !']}
    meteor = BLEUEvaluator(n=4)
    score = meteor.compute_score(ref, hypo, 'every')
    print(len(score))
    for val in score:
        print(val)


if __name__ == '__main__':
    main()
