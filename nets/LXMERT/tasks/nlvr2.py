# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.nlvr2_model import NLVR2Model
from tasks.nlvr2_data import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = NLVR2Dataset(splits)
    tset = NLVR2TorchDataset(dset)
    evaluator = NLVR2Evaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class NLVR2:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = NLVR2Model()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        self.model = self.model.cuda()

        # Losses and optimizer
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label) in iter_wrapper(enumerate(loader)):
                self.model.train()

                self.optim.zero_grad()
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)

                loss = self.mce_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    nlvr2 = NLVR2()

    # Load Model
    if args.load is not None:
        nlvr2.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'hidden' in args.test:
            nlvr2.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'hidden_predict.csv')
            )
        elif 'test' in args.test or 'valid' in args.test:
            result = nlvr2.evaluate(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, '%s_predict.csv' % args.test)
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', nlvr2.train_tuple.dataset.splits)
        if nlvr2.valid_tuple is not None:
            print('Splits in Valid data:', nlvr2.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)


