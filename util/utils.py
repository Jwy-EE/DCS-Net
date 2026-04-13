import os
import random
import numpy as np
import torch

def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_exp_settings(logger, cfg):
    pass
