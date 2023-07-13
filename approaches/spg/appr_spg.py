from typing import *

from approaches.abst_appr import AbstractAppr
from approaches.spg.model_spg import ModelSPG


class Appr(AbstractAppr):
    def __init__(self, appr_args: Dict[str, Any]):
        super().__init__(**appr_args)

        self.model = ModelSPG(**appr_args).to(self.device)
    # enddef

    def complete_learning(self, idx_task: int, **kwargs) -> None:
        dl = kwargs['dl_train']

        self.model.compute_importance(idx_task=idx_task, dl=dl)
    # enddef

    def modify_grads(self, args: Dict[str, Any]):
        self.model.modify_grads(args=args)
    # enddef
