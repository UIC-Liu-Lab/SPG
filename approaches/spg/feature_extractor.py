from typing import *

import numpy as np
from torch import Tensor, nn

from approaches.spg.spg import SPG
from utils import assert_type


class ModelAlexnet(nn.Module):
    def __init__(self, inputsize: Tuple[int, ...], nhid: int, drop1: float, drop2: float):
        super().__init__()

        nch, size = inputsize[0], inputsize[1]

        self.c1 = SPG(nn.Conv2d(nch, 64, kernel_size=size // 8))
        s = self.compute_conv_output_size(size, size // 8)
        s = s // 2

        self.c2 = SPG(nn.Conv2d(64, 128, kernel_size=size // 10))
        s = self.compute_conv_output_size(s, size // 10)
        s = s // 2

        self.c3 = SPG(nn.Conv2d(128, 256, kernel_size=2))
        s = self.compute_conv_output_size(s, 2)
        s = s // 2

        self.smid = s
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)

        self.fc1 = SPG(nn.Linear(256 * self.smid ** 2, nhid))
        self.fc2 = SPG(nn.Linear(nhid, nhid))
    # endddef

    @staticmethod
    def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1) -> int:
        return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))
    # enddef

    def forward(self, x: Tensor, args: Dict[str, Any]) -> Tuple[Tensor, Dict]:
        assert_type(x, Tensor)

        self.device = x.device

        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))

        h = h.view(h.shape[0], -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))

        misc = {
            'reg': 0
            }

        return h, misc
    # enddef

    def modify_grads(self, args: Dict[str, Any]):
        idx_task = args['idx_task']

        if idx_task == 0:
            return
        # endif

        for name_module, module in self.named_modules():
            if isinstance(module, SPG):
                module.softmask(idx_task)
            # endif
        # endfor
    # enddef
