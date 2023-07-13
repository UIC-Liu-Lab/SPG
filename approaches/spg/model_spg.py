from typing import *

from torch import Tensor, nn
from torch.utils.data import DataLoader

from approaches.spg.classifier import SPGClassifier
from approaches.spg.feature_extractor import ModelAlexnet
from approaches.spg.other_tasks_loss import OtherTasksLoss
from approaches.spg.spg import SPG
from utils import assert_type


class ModelSPG(nn.Module):
    def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...], backbone: str, nhid: int, **kwargs):
        super().__init__()

        self.device = device

        if backbone == 'alexnet':
            drop1 = kwargs['drop1']
            drop2 = kwargs['drop2']

            self.feature_extractor = ModelAlexnet(inputsize, nhid=nhid, drop1=drop1, drop2=drop2)
            self.classifier = SPGClassifier(list__ncls, dim=nhid,
                                            list__spg=[self.feature_extractor.c1, self.feature_extractor.c2, self.feature_extractor.c3,
                                                       self.feature_extractor.fc1, self.feature_extractor.fc2])
        else:
            raise NotImplementedError
        # endif
    # enddef

    def compute_importance(self, idx_task: int, dl: DataLoader):
        range_tasks = range(idx_task + 1)

        self.train()
        for t in range_tasks:
            self.zero_grad()

            for x, y in dl:
                x = x.to(self.device)
                y = y.to(self.device)

                args = {
                    'idx_task': t,
                    }

                out, _ = self.__call__(x, args=args)
                if t == idx_task:
                    lossfunc = nn.CrossEntropyLoss()
                else:
                    lossfunc = OtherTasksLoss()
                # endif

                loss = lossfunc(out, y)
                loss.backward()
            # endfor

            for name_module, module in self.named_modules():
                if isinstance(module, SPG):
                    grads = {}

                    for name_param, param in module.target_module.named_parameters():
                        if param.grad is not None:
                            grad = param.grad.data.clone().cpu()
                        else:
                            grad = 0
                        # endif

                        if name_param not in grads.keys():
                            grads[name_param] = 0
                        # endif

                        grads[name_param] += grad
                    # endfor

                    module.register_grad(idx_task=idx_task, t=t, grads=grads)
                # endif
            # endfor
        # endfor

        for name, module in self.named_modules():
            if isinstance(module, SPG):
                module.compute_mask(idx_task=idx_task)
            # endif
        # endfor
    # enddef

    def forward(self, x: Tensor, args: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        assert_type(x, Tensor)

        out, misc = self.feature_extractor(x, args=args)
        out = self.classifier(out, args=args)

        return out, misc
    # enddef

    def modify_grads(self, args: Dict[str, Any]):
        self.feature_extractor.modify_grads(args=args)
        self.classifier.modify_grads(args=args)
    # enddef
