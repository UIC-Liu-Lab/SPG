import time
from copy import deepcopy
from typing import *

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

import utils
from utils import myprint as print


class AbstractAppr:
    def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...],
                 lr: float, lr_factor: float, lr_min: float, epochs_max: int, patience_max: int,
                 lamb: float, **kwargs):
        self.device = device

        # dataloader
        self.list__ncls = list__ncls
        self.inputsize = inputsize

        # variables
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.epochs_max = epochs_max
        self.patience_max = patience_max
        self.lamb = lamb

        # misc
        self.criterion = nn.CrossEntropyLoss()
        self.model = NotImplemented  # type: nn.Module
    # enddef

    def compute_loss(self, output: Tensor, target: Tensor, misc: Dict[str, Any]) -> Tensor:
        reg = misc['reg']

        loss_all = self.criterion(output, target) + self.lamb * reg

        return loss_all
    # enddef

    def train(self, idx_task: int, dl_train: DataLoader, dl_val: DataLoader) -> Dict[str, float]:
        # optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=1.0 / self.lr_factor,
                                                         patience=max(self.patience_max - 1, 0),
                                                         min_lr=self.lr_min,
                                                         verbose=True,
                                                         )
        # metric
        patience = 0
        loss_val_best = np.inf
        acc_val_best = -np.inf
        loss_train_best = np.inf
        acc_train_best = -np.inf
        state_dict_best = self.copy_model()
        epoch_best = 0
        time_start = time.time()

        for epoch in range(self.epochs_max):
            results_train = self.train_epoch(epoch=epoch, optimizer=optimizer, idx_task=idx_task,
                                             dl_train=dl_train, dl_val=dl_val)
            loss_train, acc_train = results_train['loss_train'], results_train['acc_train']
            loss_val, acc_val = results_train['loss_val'], results_train['acc_val']

            improving = loss_val < loss_val_best
            lr_curr = utils.get_current_lr(optimizer)

            if improving:
                loss_train_best = loss_train
                acc_train_best = acc_train
                loss_val_best = loss_val
                acc_val_best = acc_val
                epoch_best = epoch
                state_dict_best = self.copy_model()
                patience = 0
            else:
                if lr_curr <= self.lr_min:
                    patience += 1
                else:
                    patience = 0
                # endif
            # endif

            if True:
                mark = '*' if improving else ' '
                msg = ' '.join([f'epoch: {epoch}/{self.epochs_max}, patience: {patience}/{self.patience_max}',
                                f'[train] loss: {loss_train_best:.4f}, acc: {acc_train_best:.4f}',
                                f'[val] loss: {loss_val_best:.4f}, acc: {acc_val_best:.4f}',
                                ])
                print(f'[{mark}] {msg}')
            # endif

            # early stop
            if patience >= self.patience_max or epoch == (self.epochs_max - 1):
                print(f'Load back to epoch={epoch_best}(loss: {loss_val_best:.4f}, acc: {acc_val_best:.4f})')
                self.load_model(state_dict_best)

                break
            # endif

            scheduler.step(loss_val)

            if np.isnan(loss_val) or np.isnan(loss_train):
                self.load_model(state_dict_best)
                print(f'Loaded model at epoch={epoch_best}')
            # endif
        # endfor

        results = {
            'epoch': epoch,
            'time_consumed': time.time() - time_start,
            'loss_train': loss_train_best,
            'acc_train': acc_train_best,
            'loss_val': loss_val_best,
            'acc_val': acc_val_best,
            }

        return results
    # enddef

    def complete_learning(self, idx_task: int, **kwargs) -> None:
        raise NotImplementedError
    # enddef

    def modify_grads(self, args: Dict[str, Any]):
        pass
    # enddef

    def train_epoch(self, epoch: int, optimizer: optim.Optimizer, idx_task: int, dl_train: DataLoader, dl_val: DataLoader) -> Dict[str, float]:
        self.model.train()
        list__target_train, list__output_train = [], []
        loss_train = 0  # type: Tensor

        for idx_batch, (x, y) in enumerate(dl_train):
            x = x.to(self.device)
            y = y.to(self.device)

            args = {
                'epoch': epoch,
                'idx_task': idx_task,
                'idx_batch': idx_batch,
                }

            output, misc = self.model(x, args=args)
            loss = self.compute_loss(output=output, target=y, misc=misc)
            loss_train += loss
            list__target_train.append(y)
            list__output_train.append(output)

            # optim
            optimizer.zero_grad()
            loss.backward()
            self.modify_grads(args)
            optimizer.step()
        # endfor

        acc_train = utils.my_accuracy(torch.cat(list__target_train, dim=0),
                                      torch.cat(list__output_train, dim=0)).item()

        # val
        results_val = self._eval_common(idx_task, dl_val)

        results = {
            'loss_train': loss_train.item(),
            'acc_train': acc_train,
            'loss_val': results_val['loss'],
            'acc_val': results_val['acc'],
            }

        return results
    # enddef

    def test(self, idx_task: int, dl_test: DataLoader) -> Dict[str, float]:
        results_test = self._eval_common(idx_task, dl_test)

        results = {
            'loss_test': results_test['loss'],
            'acc_test': results_test['acc'],
            }

        return results
    # enddef

    def _eval_common(self, idx_task: int, dl: DataLoader) -> Dict[str, float]:
        self.model.eval()
        list__target, list__output = [], []
        loss = 0  # type: Tensor

        args = {
            'idx_task': idx_task,
            }

        with torch.no_grad():
            for idx_batch, (x, y) in enumerate(dl):
                x = x.to(self.device)
                y = y.to(self.device)

                output, misc = self.model(x, args=args)
                loss += self.compute_loss(output=output, target=y, misc=misc)
                list__target.append(y)
                list__output.append(output)
            # endfor
        # endwith

        acc = utils.my_accuracy(torch.cat(list__target, dim=0),
                                torch.cat(list__output, dim=0)).item()

        results = {
            'loss': loss.item(),
            'acc': acc,
            }

        return results
    # enddef

    def copy_model(self) -> Dict[str, Tensor]:
        return deepcopy(self.model.state_dict())
    # enddef

    def load_model(self, state_dict: Dict[str, Tensor]) -> None:
        self.model.load_state_dict(state_dict)
    # enddef
