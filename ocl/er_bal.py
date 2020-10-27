from .er import *
from collections import defaultdict
from nltk.corpus import stopwords
from numpy import ma

log_dic = {}

def fastloga(arr):
    if type(arr) in [int, float]:
        x = arr
        if x not in log_dic:
            y = ma.log(x)
            log_dic[x] = y
        return log_dic[x]

    res = np.zeros_like(arr)
    for i in range(len(arr)):
        x = arr[i]
        if x not in log_dic:
            y = ma.log(x)
            log_dic[x] = y
        res[i] = log_dic[x]
    return res


class ExperienceReplayBalanced(ExperienceReplay):
    def __init__(self, base, optimizer, cfg):
        super().__init__(base, optimizer, cfg)

        self.tokenizier = self.net.tokenizer
        self.stopwords = self.tokenizier.convert_tokens_to_ids(stopwords.words('english'))

        self.mem_freqs_sum = 0
        self.data_freqs = np.zeros((32000,))
        self.data_freqs_sum = 0
        self.mem_freqs = np.zeros((32000,))
        self.word_values = np.zeros((32000,))
        self.word_forgets = np.zeros((32000,))
        self.word_forgets_n = np.zeros((32000,), dtype=np.int32)
        self.global_average_forget = 0
        self.global_average_forget_n = 0
        self.visited_words = 0

        self.smooth = get_config_attr(base.cfg, 'EXTERNAL.OCL.SMOOTH', default=1.)

        self.w2x = defaultdict(set)  # map a word to a instance id
        self.x2m = {}  # map instance id to mem position
        self.m2x = {}  # map mem position to instance id

        self.x_forgetting = {}  # map xid to their forgetting
        self.x_best_loss = {}  # map xid to the best loss
        self.x_last_visit = {}

        # variant is 'bal' or 'fgt' or 'sqrt'
        self.variant = get_config_attr(base.cfg, 'EXTERNAL.OCL.ERB_VARIANT', default='bal')
        self.length_norm = get_config_attr(base.cfg, 'EXTERNAL.OCL.LENGTH_NORM', default=0)
        self.allow_sw = get_config_attr(base.cfg, 'EXTERNAL.OCL.ALLOW_SW', default=0)

        self.kl_base_dict = {}
        self.kl_base_clean = False

        if self.allow_sw:
            self.stopwords = []
        self.reset_mem()

    def reset_mem(self):
        self.reservoir = {'x': np.zeros((self.mem_limit, self.input_size)),
                          'y': [None] * self.mem_limit,
                          'y_extra': [None] * self.mem_limit,
                          'x_value': np.zeros((self.mem_limit,)),
                          'x_tokens': [None] * self.mem_limit
                          }
        self.example_seen = 0

    def update_mem(self, *args, **kwargs):
        return self.update_values_of_examples_stream(*args, **kwargs)

    def convert_to_mem_types(self, x, y, y_extra):
        x = x.cpu().numpy()
        if type(y) not in [list, tuple]:
            y = y_to_np(y)
        else:
            y = y_to_cpu(y)
            # additionally put y[-1] to numpy
            y[-1] = y[-1].cpu().numpy()

        if type(y_extra) not in [list, tuple]:
            y_extra = y_to_np(y_extra)
        elif y_extra is not None:
            y_extra = y_to_cpu(y_extra)

        return x, y, y_extra

    def get_w2m(self, w):
        if w in self.stopwords:
            return []

        xids = self.w2x[w]
        removed = []
        ret = []
        for xid in xids:
            if xid not in self.x2m:
                removed.append(xid)
            else:
                mid = self.x2m[xid]
                xid2 = self.m2x[mid]
                if xid != xid2:  # dirty
                    self.x2m.pop(xid)
                else:
                    ret.append(mid)
        for rm in removed:
            self.w2x[w].remove(rm)
            if rm in self.x_best_loss:
                self.x_best_loss.pop(rm)
        return ret

    def get_tokens(self, y):
        label = y[-1]
        words = []
        flg = False
        for t in range(len(label)):
            word = label[t].item()
            if word != -1:
                words.append(word)
                flg = True
            elif flg:
                break
        words = [_ for _ in words if _ not in self.stopwords]
        return words

    def evaluate_value(self, tokens, base=1):
        s, cnt = 0, 0
        for x in tokens:
            if x not in self.stopwords:
                # word_weight = np.sqrt(self.data_freqs[x])
                if self.variant == 'abl':
                    word_weight = self.data_freqs[x]
                elif self.variant == 'bal':
                    word_weight = self.data_freqs[x] * (1 + self.word_forgets[x])
                elif self.variant == 'sqrt':
                    word_weight = np.sqrt(self.data_freqs[x])

                s += word_weight * (
                            np.log(self.mem_freqs[x] + self.smooth + base) - np.log(self.mem_freqs[x] + base))  # smooth
                cnt += 1
        if self.length_norm:
            s = s / (cnt + 1e-10)
        return s

    def evaluate_value_mid_updates(self, mid, tokens):
        y = self.reservoir['y'][mid]
        mem_tokens = self.get_tokens(y)
        counts = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for token in mem_tokens:
            counts[token] -= 1
        s, cnt = 0, 0
        for x in counts:
            if x not in self.stopwords:
                # word_weight = np.sqrt(self.data_freqs[x])
                if self.variant == 'abl':
                    word_weight = self.data_freqs[x]
                elif self.variant == 'bal':
                    word_weight = self.data_freqs[x] * (1 + self.word_forgets[x])
                elif self.variant == 'sqrt':
                    word_weight = np.sqrt(self.data_freqs[x])
                s += word_weight * (np.log(self.mem_freqs[x] + self.smooth + counts[x]) -
                                    np.log(self.mem_freqs[x] + self.smooth))
                cnt += 1
        if self.length_norm:
            s = s / (cnt + 1e-10)
        return s  # value increase

    def evaluate_value_change_if_swapped_and_decide_pq(self, tokens_a, all_tokens, desired):
        if not self.kl_base_clean:
            #print(tokens_a, all_tokens[0], len(all_tokens))
            self.kl_base_dict = {}
            self.kl_base_clean = True

        vocab_size = len(self.mem_freqs)

        def get_base(li):
            if li not in self.kl_base_dict:
                kl_base_b = desired * (fastloga(self.mem_freqs_sum) - fastloga(self.mem_freqs_sum + li))
                self.kl_base_dict[li] = kl_base_b.sum()
            return self.kl_base_dict[li]

        gain = []

        kbs, cs, ls, ks, ns, ds = [], [], [], [], [], []
        cnts = []
        s = 0

        for tokens_b in all_tokens:
            # kl_base_a = get_base(self.mem_freqs_sum - len(tokens_b) + len(tokens_a))
            # kl_base_b = get_base(self.mem_freqs_sum)
            # kl_change = kl_base_a - kl_base_b
            base = get_base(-len(tokens_b) + len(tokens_a))
            changes = defaultdict(int)
            for word in tokens_a:
                changes[word] += 1
            for word in tokens_b:
                changes[word] -= 1
            for word in changes:
                if changes[word] != 0:
                    c = self.mem_freqs[word]
                    l = -len(tokens_b) + len(tokens_a)
                    k = changes[word]
                    n = self.mem_freqs_sum
                    # kl_change += (c + k) / (n + l) * np.log((c + k) / (n + l)) \
                    #             - c / (n + l) * np.log(c / (n + l)) \
                    #             + k / (n + l) * np.log(desired[word])
                    cs.append(c)
                    ls.append(l)
                    ks.append(k)
                    ns.append(n)
                    ds.append(desired[word])
                    s += 1
            kbs.append(base)
            # s += len(changes)
            cnts.append(s)
            # gain.append(kl_change)

        cs, ls, ks, ns, ds = np.array(cs), np.array(ls), np.array(ks), np.array(ns), np.array(ds)
        kl_changes = ds * (fastloga(cs + ks + 1) - fastloga(cs + 1))
        prev_cnt = 0
        for i, cnt in enumerate(cnts):
            if prev_cnt == cnt:
                gain.append(0)
            else:
                gain.append(kl_changes[prev_cnt: cnt].sum() + kbs[i])
            prev_cnt = cnt

        gain = np.array(gain)

        mem_idx, value = np.argmax(gain), np.max(gain)
        # print(len(gain), mem_idx, value, gain, gain.shape)
        if value > 0:
            return mem_idx
        else:
            return -1

    def evaluate_value_mid(self, mid):
        y = self.reservoir['y'][mid]
        tokens = self.get_tokens(y)
        value = self.evaluate_value(tokens, base=0)
        return value

    def find_min_exclude(self, mids, arr):
        idx, v = 0, arr[0]
        for i in range(len(arr)):
            if arr[i] < v and i not in mids:
                idx, v = i, arr[i]
        return idx, v

    def update_values_of_examples_stream(self, x, y, y_extra=None, x_loss=None, **kwargs):
        """
        Each time we receive a stream example, we updates related values
        :param x:
        :param y:
        :param y_extra:
        :return:
        """
        if self.example_seen == 0 and self.reservoir['x'].shape[-1] != x.shape[-1]:
            self.reinit_mem(x.shape[-1])
        x, y, y_extra = self.convert_to_mem_types(x, y, y_extra)
        words = self.get_tokens(y)

        for word in words:
            if word not in self.stopwords:
                self.data_freqs[word] += 1
                self.data_freqs_sum += 1

        replaced_tokens = []
        if self.example_seen < self.mem_limit:
            j = self.example_seen
        else:
            if self.variant == 'abl':
                desired = self.data_freqs / np.sum(self.data_freqs)
            elif self.variant == 'sqrt':
                desired = np.sqrt(self.data_freqs)
                desired = desired / np.sum(desired)
            elif self.variant == 'bal':
                desired = self.data_freqs * (1 + self.word_forgets)
                desired = desired / np.sum(desired)

            mem_idx = self.evaluate_value_change_if_swapped_and_decide_pq(words, self.reservoir['x_tokens'], desired)
            j = mem_idx if mem_idx != -1 else -1
            if mem_idx != -1:
                replaced_tokens = self.get_tokens(self.reservoir['y'][j])

        if j != -1:
            added_tokens = words
            # y[-1] = torch.from_numpy(y[-1])
            self.reservoir['x'][j] = x
            self.reservoir['y'][j] = y
            self.reservoir['y_extra'][j] = y_extra
            self.reservoir['x_tokens'][j] = words
            # update mem frequency
            for token in replaced_tokens:
                if token not in self.stopwords:
                    self.mem_freqs[token] -= 1
                    self.mem_freqs_sum -= 1
            for token in added_tokens:
                if token not in self.stopwords:
                    self.mem_freqs[token] += 1
                    self.mem_freqs_sum += 1
            #for token in replaced_tokens + added_tokens:
            #    # update values
            #    mids = self.get_w2m(token)
            #    for mid in mids:
            #        self.reservoir['x_value'][mid] = self.evaluate_value_mid(mid)
            xid = self.example_seen

            self.m2x[j] = xid
            self.x2m[xid] = j
            for token in added_tokens:
                if token not in self.stopwords:
                    self.w2x[token].add(xid)
            self.x_best_loss[xid] = x_loss  # [T]
            self.kl_base_clean = False

        self.example_seen += 1


    def observe(self, x, y, task_ids=None, extra=None, optimize=True):
        # recover image, feat from x

        (mem_x, mem_indices), mem_y = self.sample_mem_batch(x.device, return_indices=True)

        batch_size = x.size(0)

        combined_x, combined_y = x, y

        ret_dict = self.forward_net(combined_x, combined_y, reduce=False)

        loss_mat = ret_dict['loss'].view(x.size(0), -1)  # [B,T]
        loss_mat_np_x = loss_mat_np = loss_mat.detach().cpu().numpy()
        mask_cnts = torch.FloatTensor(ret_dict['mask_cnts']).unsqueeze(-1).to(x.device)
        mask_cnts = mask_cnts + 1e-10
        loss_reduced = loss_mat / mask_cnts
        ret_dict['loss'] = loss_reduced.sum(-1).mean()

        if optimize:
            # loss = loss_tmp.mean()
            # print(loss.item())
            if self.concat_replay or self.separate_replay:
                loss = ret_dict['loss']
            else:
                loss_tmp = ret_dict['loss']
                loss = loss_tmp[: x.size(0)].mean()
                if mem_x is not None:
                    loss += loss_tmp[x.size(0):].mean()

            self.optimizer.zero_grad()

            if self.concat_replay and mem_x is not None:
                loss = loss / 2

            loss.backward()

            if not self.concat_replay or mem_x is None:
                self.optimizer.step()

            if self.separate_replay  and mem_x is not None:
                ret_dict_mem = self.forward_net(mem_x, mem_y, reduce=False)

                loss_mat = ret_dict_mem['loss'].view(mem_x.size(0), -1) # [B,T]

                loss_mat_np = loss_mat.detach().cpu().numpy()
                gt_label = mem_y[-1].cpu().numpy() # [B,T]

                for b, mem_idx in enumerate(mem_indices):
                    flg = False
                    xid = self.m2x[mem_idx]
                    best_loss = self.x_best_loss[xid]

                    # update best loss for each word
                    for t in range(min(len(best_loss), len(loss_mat_np[b]))):
                        if loss_mat_np[b,t] < best_loss[t]:
                            self.x_best_loss[xid][t] = loss_mat_np[b,t]

                    for t in range(loss_mat.size(1)):
                        word = gt_label[b,t]
                        if word != -1:
                            loss_word = loss_mat_np[b,t]
                            # if loss_word < self.x_best_loss[xid]:
                            #     self.x_best_loss[xid] = loss_word
                            forget = loss_word - self.x_best_loss[xid][t]

                            # move average
                            if self.word_forgets_n[word] == 0:
                                self.word_forgets[word] = self.global_average_forget
                                self.word_forgets_n[word] = self.global_average_forget_n / (1e-10 + self.visited_words)
                                self.visited_words += 1
                            else:
                                self.word_forgets[word] = (self.word_forgets[word] * self.word_forgets_n[word] + forget) / \
                                                      (self.word_forgets_n[word] + 1)
                            self.word_forgets_n[word] += 1

                            self.global_average_forget = (self.global_average_forget * self.global_average_forget_n + forget) / \
                                                         (self.global_average_forget_n + 1)
                            self.global_average_forget_n += 1

                            flg = True
                        elif flg:
                            break
                        self.x_last_visit[word] = self.example_seen

                mask_cnts = torch.FloatTensor(ret_dict_mem['mask_cnts']).unsqueeze(-1).to(x.device)
                mask_cnts = mask_cnts + 1e-10

                loss_reduced = loss_mat / mask_cnts
                loss_reduced = loss_reduced.sum(-1).mean()

                # stat forgetting

                if not self.concat_replay:
                    self.optimizer.zero_grad()

                if self.concat_replay:
                    loss_reduced = loss_reduced / 2

                loss_reduced.backward()
                self.optimizer.step()
                ret_dict['loss'] = (ret_dict['loss'] + loss_reduced) / 2

            for b in range(batch_size):  # x.size(0)
                if type(y) is tuple:
                    y_ = [_[b] for _ in y]
                else:
                    y_ = y[b]
                self.update_mem(x[b], y_, x_loss=loss_mat_np_x[b])
        ret_dict['loss'] = ret_dict['loss'].detach().cpu()
        return ret_dict


