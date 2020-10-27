from collections import Counter
import uuid, time, datetime, os, torch, logging
from collections import OrderedDict
import numpy as np
import errno
import random

def get_top_k_by_frequency(vocab, top_k=None):
    '''
    Get the top k elements from a list by frequency
    :param vocab: the list
    :parm top_k: top_k
    :return: top k elements
    '''
    counter = Counter(vocab)

    sorted_vocab = sorted(
        [t for t in counter],
        key=counter.get,
        reverse=True
    )

    if top_k:
        return sorted_vocab[:top_k]
    
    return sorted_vocab

def get_exp_id():
    return uuid.uuid4().hex[:6]


class Timer(object):
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.add(time.time() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self):
        time_str = str(datetime.timedelta(seconds=self.average_time))
        return time_str

    class Checkpointer(object):
        def __init__(
                self,
                model,
                optimizer=None,
                scheduler=None,
                save_dir="",
                save_to_disk=None,
                logger=None,
        ):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.save_dir = save_dir
            self.save_to_disk = save_to_disk
            if logger is None:
                logger = logging.getLogger(__name__)
            self.logger = logger

        def save(self, name, **kwargs):
            if not self.save_dir:
                return

            if not self.save_to_disk:
                return

            data = {}
            data["model"] = self.model.state_dict()
            if self.optimizer is not None:
                data["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
            data.update(kwargs)

            save_file = os.path.join(self.save_dir, "{}.pth".format(name))
            self.logger.info("Saving checkpoint to {}".format(save_file))
            torch.save(data, save_file)
            self.tag_last_checkpoint(save_file)

        def load(self, f=None, use_latest=True):
            if self.has_checkpoint() and use_latest:
                # override argument with existing checkpoint
                f = self.get_checkpoint_file()
            if not f:
                # no checkpoint could be found
                self.logger.info("No checkpoint found. Initializing model from scratch")
                return {}
            self.logger.info("Loading checkpoint from {}".format(f))
            checkpoint = self._load_file(f)
            self._load_model(checkpoint)
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

            # return any further checkpoint data
            return checkpoint

        def has_checkpoint(self):
            save_file = os.path.join(self.save_dir, "last_checkpoint")
            return os.path.exists(save_file)

        def get_checkpoint_file(self):
            save_file = os.path.join(self.save_dir, "last_checkpoint")
            try:
                with open(save_file, "r") as f:
                    last_saved = f.read()
                    last_saved = last_saved.strip()
            except IOError:
                # if file doesn't exist, maybe because it has just been
                # deleted by a separate process
                last_saved = ""
            return last_saved

        def tag_last_checkpoint(self, last_filename):
            save_file = os.path.join(self.save_dir, "last_checkpoint")
            with open(save_file, "w") as f:
                f.write(last_filename)

        def _load_file(self, f):
            return torch.load(f, map_location=torch.device("cpu"))

        def _load_model(self, checkpoint):
            load_state_dict(self.model, checkpoint.pop("model"))

def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict)

def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def get_config_attr(cfg, attr_string, default=None, totype=None, mute=False):
    try:
        attrs = attr_string.split('.')
        obj = cfg
        for s in attrs:
            obj = getattr(obj, s)
        if totype is None:
            return type(default)(obj) if default is not None else obj
        else:
            if totype is bool and obj not in ['True','False',True,False]:
                raise ValueError('malformed boolean input: {}, {}'.format(obj,type(obj)))
            if totype is bool and obj in ['False',False]:
                return False
            return totype(obj)
    except AttributeError:
        #if not mute:
        #    print('Warning: attribute {} not found. Default: {}'.format(attr_string, default))
        return default

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, **dct):
        for key, value in dct.items():
            #if hasattr(value, 'keys'):
            #    value = DotDict(**value)
            self[key] = value


def set_config_attr(cfg, attr_key, attr_value):
    attrs = attr_key.split('.')
    obj = cfg
    for attr in attrs[:-1]:
        if not hasattr(obj, attr):
            d = DotDict()
            setattr(obj, attr, d)
        obj = getattr(obj, attr)

    try:
        attr_value = int(attr_value)
    except ValueError:
        try:
            attr_value = float(attr_value)
        except ValueError:
            pass
            pass

    if attr_value == 'True': attr_value = True
    if attr_value == 'False': attr_value = False

    setattr(obj, attrs[-1], attr_value)



def filter_outliers(l):
    q1 = np.quantile(l, 0.25)
    q2 = np.quantile(l, 0.75)
    iqr = q2 - q1
    lb, rb = q1 - 1.5 * iqr, q2 + 1.5 * iqr
    l = [x for x in l if lb <= x <= rb]
    return l

def set_cfg_from_args(args, cfg):
    cfg_params = args.cfg
    if cfg_params is None: return
    for param in cfg_params:
        k, v = param.split('=')
        set_config_attr(cfg, k, v)


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class CheckpointerFromCfg(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super().__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_config(cfg, path):
    with open(path, 'w') as f:
        f.write(cfg.dump())

def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

