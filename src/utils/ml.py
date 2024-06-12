import numpy as np
import random
import torch


def seed_everything(seed=3407):
    #os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed%(2**32-1))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def get_dtype(mix_precision_type):
    if mix_precision_type == str(16):
        return torch.float16
    elif mix_precision_type == str(32):
        return torch.float32
    elif mix_precision_type == "bf16":
        return torch.bfloat16
    else:
        raise ValueError("Unsupported mix_precision_type")