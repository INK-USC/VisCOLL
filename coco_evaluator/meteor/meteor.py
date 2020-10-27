# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class METEOREvaluator:
    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, encoding='utf-8', bufsize=0)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res, mode='all'):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert (len(res[i]) == 1)
            stat, score_line = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        for i in range(0, len(imgIds)):
            score = self.meteor_p.stdout.readline().strip()
            scores.append(float(score))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        if mode == 'all':
            return [score]
        elif mode == 'every':
            return scores
        else:
            return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        out = self.meteor_p.stdout.readline().strip()
        return out, score_line

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()


def main():
    hypo = {'1': ['I like it.i Love You..next'], '2': ['I completely do not know !']}
    ref = {'1': ['I love you !'], '2': ['I do not know !']}
    for i in range(30):
        hypo[i] = ['The dismantling of the Punggye-ri site, the exact date of which will depend on ' \
                  'weather conditions, will involve the collapsing of all tunnels using explosives ' \
                  'and the removal of all observation facilities, ' \
                  'research buildings and security posts. Journalists from South Korea, China, the US, ' \
                  'the UK and Russia will be asked to attend to witness the event.' \
                  'North Korea said the intention was to allow ' \
                  'not only the local press but also journalists of other countries to conduct on-the-spot coverage ' \
                  'in order to show in a transparent manner the dismantlement of the northern nuclear test ground.'\
                  'The reason officials gave for limiting the number of countries invited to send journalists ' \
                  'was due to the small space of the test ground... located in the uninhabited deep mountain area.']
        ref[i] = ['There is a "sense of optimism" among North Korea\'s leaders, the head of the UN\'s ' \
                 'World Food Programme (WFP) said on Saturday after enjoying what he said was ' \
                 'unprecedented access to the country. David Beasley spent two days in the capital, Pyongyang, ' \
                 'and two outside it, accompanied by government minders. He said the country was working hard ' \
                 'to meet nutritional standards, and hunger was not as high as in the 1990s. Mr Beasley\'s visit, ' \
                 'from 8-11 May, included trips to WFP-funded projects - a children\'s ' \
                 'nursery in South Hwanghae province and a fortified biscuit factory in North North Pyongyan province.']
    meteor = METEOREvaluator()
    score = meteor.compute_score(ref, hypo, 'every')
    print(score)
    pass


if __name__ == '__main__':
    main()
