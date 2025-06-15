
from functools import wraps

import torch
from torch.utils.data import default_collate


def list_in_list_out(f):
    @wraps(f)
    def decorated(self, data, *args, **kwargs):
        
        if isinstance(data, torch.Tensor):
            return f(self, data, *args, **kwargs)
        
        out_list = []
        for sample in data:
            one_sample_batch = default_collate([sample])
            cur_out = f(self, one_sample_batch, *args, **kwargs)
            out_list.append(cur_out[0])

        return out_list

    return decorated