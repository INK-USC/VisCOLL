import torch
import numpy as np
import pickle
from utils.utils import get_config_attr
import copy
from .naive import NaiveWrapper
from torch.optim import SGD, Adam
import math
try:
    from pytorch_transformers import AdamW
except ImportError:
    AdamW = None


def y_to_np(y):
    if type(y) is tuple:
        return tuple(x.item() for x in y)
    else:
        return y.cpu().numpy()


def y_to_cpu(y):
    if torch.is_tensor(y):
        y = y.cpu()
    else:
        y = [_.cpu() for _ in y]
    return y


def index_select(l, indices, device):
    ret = []
    for i in indices:
        if type(l[i]) is np.ndarray:
            x = torch.from_numpy(l[i]).to(device)
            ret.append(x.unsqueeze(0))
        else:
            if type(l[i]) is list:
                item = []
                for j in range(len(l[i])):
                    if type(l[i][j]) is np.ndarray:
                        item.append(torch.from_numpy(l[i][j]))
                    else:
                        item.append(l[i][j])
                ret.append(item)
            else:
                ret.append(l[i])
    return ret


def concat_with_padding(l):
    if l is None or l[0] is None: return None
    if type(l[0]) in [list, tuple]:
        ret = [torch.cat(t, 0) for t in zip(*l)]
    else:
        if len(l[0].size()) == 2:  # potentially requires padding
            max_length = max([x.size(1) for x in l])
            ret = []
            for x in l:
                pad = torch.zeros(x.size(0), max_length - x.size(1)).long().to(x.device)
                x_pad = torch.cat([x, pad], -1)
                ret.append(x_pad)
            ret = torch.cat(ret, 0)
        else:
            ret = torch.cat(l, 0)
    return ret


def store_grad(pp, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1


class ExperienceReplay(NaiveWrapper):
    input_size: int

    def __init__(self, base, optimizer, cfg):
        super().__init__(base, optimizer, cfg)
        self.net = base
        self.optimizer = optimizer
        self.mem_limit = cfg.EXTERNAL.REPLAY.MEM_LIMIT
        self.mem_bs = cfg.EXTERNAL.REPLAY.MEM_BS
        self.reservoir, self.example_seen = None, None
        self.mem_occupied = {}

        self.policy = get_config_attr(cfg, 'EXTERNAL.OCL.POLICY', default='reservoir', totype=str)

        # configs about MIR
        self.mir_k = get_config_attr(cfg, 'EXTERNAL.REPLAY.MIR_K', default=10, totype=int)
        self.mir = get_config_attr(cfg, 'EXTERNAL.OCL.MIR', default=0, totype=int)
        self.mir_agg = get_config_attr(cfg, 'EXTERNAL.OCL.MIR_AGG', default='avg', totype=str)

        self.cfg = cfg
        self.grad_dims = []

        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

        self.example_seen = 0

    def reset_mem(self):
        self.reservoir = {'x': np.zeros((self.mem_limit, self.input_size)),
                          'y': [None] * self.mem_limit,
                          'y_extra': [None] * self.mem_limit
                          }
        self.example_seen = 0

    def update_mem(self, *args, **kwargs):
        self.update_mem_reservoir(*args, **kwargs)

    def reinit_mem(self, xsize):
        self.input_size = xsize
        self.reset_mem()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def update_mem_reservoir(self, x, y, loss_x=None, *args, **kwargs):
        if self.reservoir is None:
            self.reinit_mem(x.shape[-1])

        x = x.cpu().numpy()
        if type(y) not in [list, tuple]:
            y = y_to_np(y)
        else:
            y = y_to_cpu(y)
        if self.example_seen < self.mem_limit:
            self.reservoir['x'][self.example_seen] = x
            self.reservoir['y'][self.example_seen] = y
            j = self.example_seen
        else:
            j = np.random.RandomState(self.example_seen + self.cfg.SEED).randint(0, self.example_seen)
            if j < self.mem_limit:
                self.reservoir['x'][j] = x
                self.reservoir['y'][j] = y
        if loss_x is not None:
            self.reservoir['loss_stats'][j] = [loss_x]
            self.reservoir['loss_stat_steps'][j] = [self.example_seen]
            self.reservoir['forget'] = 0
        self.example_seen += 1

    def get_available_index(self):
        l = []
        for idx, t in enumerate(self.seen_tasks):
            offset_start, offset_stop = self.compute_offset(t, len(self.seen_tasks))
            for i in range(offset_start, min(offset_start + self.mem_occupied[t], offset_stop)):
                l.append(i)
        return l

    def get_random(self, seed=1):
        random_state = None
        for i in range(seed):
            if random_state is None:
                random_state = np.random.RandomState(self.example_seen + self.cfg.SEED)
            else:
                random_state = np.random.RandomState(random_state.randint(0, int(1e5)))
        return random_state

    def store_cache(self):
        self.cache = copy.deepcopy(self.net.state_dict())

    def load_cache(self):
        self.net.load_state_dict(self.cache)
        self.net.zero_grad()

    def get_loss_and_pseudo_update(self, x, y):
        ret_dict_d = self.forward_net(x, y)
        self.optimizer.zero_grad()
        ret_dict_d['loss'].backward(retain_graph=False)
        if isinstance(self.optimizer, AdamW):
            step_wo_state_update_adamw(self.optimizer)
        else:
            raise NotImplementedError
        return ret_dict_d

    def decide_mir_mem(self, x, y, mir_k, cand_x, cand_y, indices, mir_least):
        if cand_x.size(0) < mir_k:
            return cand_x, cand_y, indices
        else:
            self.store_cache()
            if type(cand_y[0]) not in [list, tuple]:
                cand_y = concat_with_padding(cand_y)
            else:
                cand_y = [torch.stack(_).to(x.device) for _ in zip(*cand_y)]
            with torch.no_grad():
                ret_dict_mem_before = self.forward_net(cand_x, cand_y, reduce=False)
            ret_dict_d = self.get_loss_and_pseudo_update(x, y)
            with torch.no_grad():
                ret_dict_mem_after = self.forward_net(cand_x, cand_y, reduce=False)
                loss_increase = ret_dict_mem_after['loss'] - ret_dict_mem_before['loss']
            with torch.no_grad():
                if self.mir_agg == 'avg':
                    loss_increase_by_ts = loss_increase.view(cand_x.size(0), -1).sum(1)
                    mask_num_by_ts = (cand_y[2] != -1).sum(1).float() + 1e-10
                    loss_increase = loss_increase_by_ts / mask_num_by_ts
                elif self.mir_agg == 'max':
                    loss_increase, _ = loss_increase.view(cand_x.size(0), -1).max(1)

                _, topi = loss_increase.topk(mir_k, largest=not mir_least)
                if type(cand_y) is not list:
                    mem_x, mem_y = cand_x[topi], cand_y[topi]
                else:
                    mem_x = cand_x[topi]
                    mem_y = [_[topi] for _ in cand_y]

            self.load_cache()
            return mem_x, mem_y, indices[topi.cpu()]

    def sample_mem_batch(self, device, return_indices=False, k=None, seed=1,
                         mir=False, input_x=None, input_y=None, mir_k=0,
                         mir_least=False):
        random_state = self.get_random(seed)
        if k is None:
            k = self.mem_bs
        # reservoir
        n_max = min(self.mem_limit, self.example_seen)
        available_indices = [_ for _ in range(n_max)]

        if not available_indices:
            if return_indices:
                return None, None
            else:
                return None, None
        elif len(available_indices) < k:
            indices = np.arange(n_max)
        else:
            indices = random_state.choice(available_indices, k, replace=False)

        x = self.reservoir['x'][indices]
        x = torch.from_numpy(x).to(device).float()

        y = index_select(self.reservoir['y'], indices, device)  # [  [...], [...] ]

        if type(y[0]) not in [list, tuple]:
            y_pad = concat_with_padding(y)
        else:
            y_pad = [torch.stack(_).to(device) for _ in zip(*y)]

        if mir:
            x, y_pad, indices = self.decide_mir_mem(input_x, input_y, mir_k,
                                                             x, y, indices, mir_least)

        if not return_indices:
            return x, y_pad
        else:
            return (x, indices), y_pad

    def observe(self, x, y, task_ids=None, extra=None, optimize=True):

        if self.mir:
            mem_x, mem_y = self.sample_mem_batch(x.device, input_x=x, input_y=y,
                                                 mir_k=self.mir_k, mir=self.mir)
        else:
            mem_x, mem_y = self.sample_mem_batch(x.device)

        batch_size = x.size(0)

        ret_dict = self.forward_net(x, y, reduce=True)

        loss = ret_dict['loss']
        if optimize:
            # main loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if mem_x is not None:
                ret_dict_mem = self.forward_net(mem_x, mem_y, reduce=True)
                self.optimizer.zero_grad()
                ret_dict_mem['loss'].backward()
                self.optimizer.step()
                ret_dict['loss'] = (ret_dict['loss'] + ret_dict_mem['loss']) / 2

            for b in range(batch_size):  # x.size(0)
                if type(y) is tuple:
                    y_ = [_[b] for _ in y]
                else:
                    y_ = y[b]
                self.update_mem(x[b], y_)
        return ret_dict

    def dump_reservoir(self, path, verbose=False):
        f = open(path, 'wb')
        pickle.dump({
            'reservoir_x': self.reservoir['x'] if verbose else None,
            'reservoir_y': self.reservoir['y'],
            'mem_occupied': self.mem_occupied,
            'example_seen': self.example_seen,
        }, f)
        f.close()

    def load_reservoir(self, path):
        try:
            f = open(path, 'rb')
            dic = pickle.load(f)
            for k in dic:
                setattr(self, k, dic[k])
            f.close()
            return dic
        except FileNotFoundError:
            print('no replay buffer dump file')
            return {}

    def load_reservoir_from_dic(self, dic):
        for k in dic:
            setattr(self, k, dic[k])

    def get_reservoir(self):
        return {'reservoir': self.reservoir, 'mem_occupied': self.mem_occupied,
                'example_seen': self.example_seen, 'seen_tasks': self.seen_tasks,
                'balanced': self.balanced}


def step_wo_state_update_adamw(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p]

            # State initialization
            #if len(state) == 0:
            #    state['step'] = 0
            #    # Exponential moving average of gradient values
            #    state['exp_avg'] = torch.zeros_like(p.data)
            #    # Exponential moving average of squared gradient values
            #    state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            #state['step'] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg = exp_avg.mul(beta1).add(1.0 - beta1, grad)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(1.0 - beta2, grad, grad)
            denom = exp_avg_sq.sqrt().add(group['eps'])

            step_size = group['lr']
            if group['correct_bias']:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg, denom)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group['weight_decay'] > 0.0:
                p.data.add_(-group['lr'] * group['weight_decay'], p.data)

    return loss