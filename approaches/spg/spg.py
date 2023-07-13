from typing import *

import torch
from torch import Tensor, nn

from utils import assert_type, myprint as print


class SPG(nn.Module):
    def __init__(self, target_module: nn.Module):
        super().__init__()
        assert_type(target_module, nn.Module)

        self.target_module = target_module

        self.history_mask = dict()  # type: Dict[int, Dict[str, Tensor]]
        self.dict_amax = {}

        self.dict__idx_task__t__h = {}  # type: Dict[int, Dict[int, Dict[str, Tensor]]]
    # enddef

    def forward(self, x: Tensor) -> Tensor:
        assert_type(x, Tensor)

        out = self.target_module(x)

        return out
    # enddef

    def standardize_pm1(self, x: Tensor) -> Tensor:
        if torch.all(x == 0):
            pass
        else:
            x = self.standardize(x)
        # endif
        ret = torch.tanh(x)

        return ret
    # enddef

    @classmethod
    def standardize(cls, x: Tensor) -> Tensor:
        sh = x.shape
        x = x.view(-1)

        ret = (x - x.mean()) / x.std()

        return ret.view(*sh)
    # enddef

    def register_grad(self, idx_task: int, t: int, grads: Dict[str, Tensor]):
        if idx_task not in self.dict__idx_task__t__h.keys():
            self.dict__idx_task__t__h[idx_task] = {}
        # endif

        if t not in self.dict__idx_task__t__h[idx_task].keys():
            self.dict__idx_task__t__h[idx_task][t] = {}
        # endif

        for name, grad in grads.items():
            if name in self.dict__idx_task__t__h[idx_task][t].keys():
                grad_prev = self.dict__idx_task__t__h[idx_task][t][name]
            else:
                grad_prev = 0
            # endif

            grad_new = grad_prev + grad

            self.dict__idx_task__t__h[idx_task][t][name] = grad_new
        # endfor
    # enddef

    def compute_mask(self, idx_task: int):
        if idx_task not in self.dict__idx_task__t__h.keys():
            # ablations can take this route.
            return
        # endif

        names = self.dict__idx_task__t__h[idx_task][idx_task].keys()
        history = {}  # type: Dict[str, Tensor]
        for t, dict__name__h in self.dict__idx_task__t__h[idx_task].items():
            assert names == dict__name__h.keys()
            for name, h in dict__name__h.items():
                if name not in history.keys():
                    history[name] = torch.zeros_like(h)
                # endif

                history[name] = torch.max(history[name], self.standardize_pm1(h).abs())
            # endfor
        # endfor

        self.history_mask = {idx_task: history.copy()}

        self.dict__idx_task__t__h.clear()
    # enddef

    def a_max(self, idx_task: int, latest_module: nn.Module) -> Dict[str, Tensor]:
        if idx_task == 0:
            return None
        else:
            if idx_task not in self.dict_amax.keys():
                ret = dict()

                for name_param, param in latest_module.named_parameters():
                    curr = self.history_mask[idx_task - 1][name_param]
                    if idx_task - 1 in self.dict_amax.keys():
                        prev = self.dict_amax[idx_task - 1][name_param]
                    else:
                        prev = curr
                    # endif

                    v1 = torch.max(prev, curr)
                    ret[name_param] = v1
                # endfor

                self.dict_amax[idx_task] = ret
            # endif

            return self.dict_amax[idx_task]
        # endif
    # enddef

    def softmask(self, idx_task: int):
        tgt = self.target_module

        a_max = self.a_max(idx_task, tgt)

        for n, p in tgt.named_parameters():
            if p.grad is None:
                pass
            else:
                red = (1 - a_max[n]).to(p.device)
                p.grad.data *= red

                if False:
                    num_0 = red[red == 0].numel()
                    num_all = red.numel()

                    classname = self.target_module.__class__.__name__
                    msg = f'[{classname}.{n}]' \
                          f' dead: {num_0}/{num_all}({num_0 / num_all:.2%})'
                    print(msg)
                # endif
            # endif
        # endfor
    # enddef
