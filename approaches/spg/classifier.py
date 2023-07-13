from typing import *

import torch
from torch import Tensor, nn

from approaches.spg.spg import SPG
from utils import assert_type


class SPGClassifier(nn.Module):
    def __init__(self, list__ncls: List[int], dim: int, list__spg: List[SPG]):
        super().__init__()

        self.list__classifier = nn.ModuleList()
        for ncls in list__ncls:
            head = _TaskHead(nn.Linear(dim, ncls), list__spg=list__spg)
            self.list__classifier.append(head)
        # endfor
    # enddef

    def forward(self, x: Tensor, args: Dict[str, Any]) -> Tensor:
        assert_type(x, Tensor)

        idx_task = args['idx_task']

        clf = self.list__classifier[idx_task]
        x = x.view(x.shape[0], -1)
        out = clf(x)

        return out
    # enddef

    def modify_grads(self, args: Dict[str, Any]) -> None:
        idx_task = args['idx_task']

        torch.nn.utils.clip_grad_norm_(self.parameters(), 10000)

        for _, module in self.list__classifier.named_modules():
            if isinstance(module, _TaskHead):
                module.softmask(idx_task=idx_task)
            # endif
        # endfor
    # enddef


class _TaskHead(nn.Module):
    def __init__(self, classifier: nn.Linear, list__spg: List[SPG]):
        super().__init__()

        self.classifier = classifier
        self.list__spg = list__spg

        self.dict__idx_task__red = {}

        self.device = None
    # enddef

    def forward(self, x: Tensor) -> Tensor:
        if self.device is None:
            self.device = x.device
        # endif

        return self.classifier(x)
    # enddef

    def softmask(self, idx_task: int):
        if idx_task not in self.dict__idx_task__red.keys():
            list__amax = []
            for spg in self.list__spg:
                dict__amax = spg.a_max(idx_task=idx_task, latest_module=spg.target_module)
                if dict__amax is not None:
                    for _, amax in dict__amax.items():
                        list__amax.append(amax.view(-1))
                    # endfor
                # endif
            # endfor

            if len(list__amax) > 0:
                amax = torch.cat(list__amax, dim=0)
                mean_amax = amax.mean()
                modification = (1 - mean_amax).cpu().item()
            else:
                modification = 1
            # endif

            self.dict__idx_task__red[idx_task] = modification
        else:
            modification = self.dict__idx_task__red[idx_task]
        # endif

        if False:
            print(f'[classifier] modification: {modification}')
        # endif

        for n, p in self.classifier.named_parameters():
            if p.grad is not None:
                p.grad.data *= modification
            # endif
        # endfor
    # enddef
