import torch
from torch import nn
from .er import ExperienceReplay, store_grad
from utils.utils import get_config_attr

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


class AGEM(ExperienceReplay):
    def __init__(self, base, optimizer, cfg):
        super(AGEM, self).__init__(base, optimizer, cfg)

        # self.grads = self.grads.cuda()
        self.violation_count = 0
        self.agem_k = get_config_attr(cfg, 'EXTERNAL.OCL.AGEM_K', default=256)

    def compute_grad(self, mem_x, mem_y):
        self.zero_grad()
        ret_dict = self.forward_net(mem_x, mem_y)
        # regularize both of the loss
        loss = ret_dict['loss']
        loss.backward()
        grads = torch.Tensor(sum(self.grad_dims)).to(mem_x.device)
        store_grad(self.parameters, grads, self.grad_dims)
        return grads

    def fix_grad(self, mem_grads):
        # check whether the current gradient interfere with the average gradients
        grads = torch.Tensor(sum(self.grad_dims)).to(mem_grads.device)
        store_grad(self.parameters, grads, self.grad_dims)
        dotp = torch.dot(grads, mem_grads)
        if dotp < 0:
            # project the grads back to the mem_grads
            # g_new = g - g^Tg_{ref} / g_{ref}^Tg_{ref} * g_{ref}
            new_grad = grads - (torch.dot(grads, mem_grads) / (torch.dot(mem_grads, mem_grads) + 1e-12)) * mem_grads
            overwrite_grad(self.parameters, new_grad, self.grad_dims)
            return 1
        else:
            return 0

    def observe(self, x, y, task_ids=None, extra=None, optimize=True):
        # recover image, feat from x
        self.optimizer.zero_grad()
        mem_x, mem_y = self.sample_mem_batch(x.device, k=self.agem_k)

        if mem_x is not None:
            # calculate gradients on the memory batch
            mem_grads = self.compute_grad(mem_x, mem_y)

        # backward on the current minibatch
        self.optimizer.zero_grad()
        batch_size = x.size(0)
        ret_dict = \
            self.forward_net(x, y)
        for b in range(batch_size):
            #self.update_mem(x[b], y[b], task_ids[b] if task_ids is not None else None)
            if type(y) is tuple:
                y_ = [_[b] for _ in y]
            else:
                y_ = y[b]
            self.update_mem(x[b], y_)
        loss = ret_dict['loss']
        if optimize:
            loss.backward()
            if mem_x is not None:
                violated = self.fix_grad(mem_grads)
                self.violation_count += violated

            self.optimizer.step()

        return ret_dict
