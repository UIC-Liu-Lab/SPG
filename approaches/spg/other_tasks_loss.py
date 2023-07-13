from torch import Tensor, nn


class OtherTasksLoss(nn.Module):
    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        return t.sum()
    # enddef
