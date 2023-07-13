import json
import os
from typing import *

import numpy as np
from torch.utils.data import DataLoader

from utils import myprint as print


class MyMetrics:
    def __init__(self, list__name: List[str], list__dl_test: List[DataLoader], sufix: str = ''):
        self.list__name = list__name
        self.num_tasks = len(list__name)
        self.list__num_test = [len(dl.dataset) for dl in list__dl_test]

        self.nda_loss = np.zeros([self.num_tasks, self.num_tasks]) * np.nan
        self.nda_acc = np.zeros([self.num_tasks, self.num_tasks]) * np.nan

        self.dict__name__idx_task = {name: [t for t in range(self.num_tasks) if list__name[t] == name]
                                     for name in set(list__name)}
        self.dict__name__idx_task['Overall'] = list(range(self.num_tasks))
        self.dict__idx_task__misc = {}  # {t: {} for t in range(self.num_tasks)}

        self.sufix = sufix
    # enddef

    def add_record(self, idx_task_learned: int, idx_task_tested: int,
                   loss: float, acc: float) -> None:
        self.nda_loss[idx_task_learned, idx_task_tested] = loss
        self.nda_acc[idx_task_learned, idx_task_tested] = acc
    # enddef

    def add_record_misc(self, idx_task_learned: int, epoch: int, time_consumed: float):
        results = {
            'epoch': epoch,
            'time_consumed': time_consumed,
            }
        print(f'idx_task: {idx_task_learned}, results: {results}')

        self.dict__idx_task__misc[idx_task_learned] = results
    # enddef

    def save(self, dir: str, idx_task_latest: int, indices_task_ignored=[]) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
        list__artifact = []

        # loss/acc
        path_loss = os.path.join(dir, f'{self.sufix}loss.csv')
        np.savetxt(path_loss, self.nda_loss, delimiter=',', header=','.join(self.list__name), comments='')
        list__artifact.append(path_loss)
        path_acc = os.path.join(dir, f'{self.sufix}acc.csv')
        np.savetxt(path_acc, self.nda_acc, delimiter=',', header=','.join(self.list__name), comments='')
        list__artifact.append(path_acc)

        # metrics
        dict__idx_task__metrics = {}
        sec_total_consumed = 0
        for idx_task in range(idx_task_latest + 1):
            dict__misc = self.dict__idx_task__misc[idx_task]
            sec_each_consumed = dict__misc['time_consumed']
            sec_total_consumed += sec_each_consumed
            epoch = dict__misc['epoch']
            metrics = {'sec_each_consumed': sec_each_consumed,
                       'sec_total_consumed': sec_total_consumed,
                       'epoch': epoch,
                       }

            for name, list__idx_task in self.dict__name__idx_task.items():
                list__idx_task = [t for t in list__idx_task if
                                  t <= idx_task and t not in indices_task_ignored]
                if len(list__idx_task) == 0:
                    acc_avg = None
                    btf = None
                else:
                    weight = np.array(self.list__num_test)[list__idx_task]

                    # avg acc
                    list__acc = self.nda_acc[idx_task, list__idx_task]
                    # acc_avg = np.mean(list__acc).item()
                    acc_avg = np.average(list__acc, weights=weight).item()
                    metrics[f'acc__{name}'] = acc_avg

                    # forward transfer
                    pass

                    # backward transfer
                    list__acc_initial = self.nda_acc.diagonal()[list__idx_task]
                    # btf = np.mean(list__acc - list__acc_initial).item()
                    btf = np.average(list__acc - list__acc_initial, weights=weight).item()
                    num_tasks = len(list__idx_task)
                    if num_tasks == 1:
                        btf = 0
                    else:
                        btf = btf * num_tasks / (num_tasks - 1)
                    # endif
                    metrics[f'btf__{name}'] = btf
                # endif
            # endfor

            dict__idx_task__metrics[idx_task] = metrics
        # endfor
        '''
        dict__idx_task__metrics = {
          0: {
            'sec_each_consumed': xxx,
            'sec_total_consumed': xxx,
            'acc__cifar100_10': xxx,
            'loss__cifar100_10': xxx,
            'acc__Overall': xxx,
            'loss__Overall': xxx,
          },
          1: {},
          ...
          9: {},
        } 
        '''

        path_metrics = os.path.join(dir, f'{self.sufix}metrics.json')
        with open(path_metrics, 'w') as fp:
            json.dump(dict__idx_task__metrics, fp)
        # endwith
        list__artifact.append(path_metrics)

        path_metrics_latest = os.path.join(dir, f'{self.sufix}metrics_latest.json')
        with open(path_metrics_latest, 'w') as fp:
            json.dump({idx_task_latest: dict__idx_task__metrics[idx_task_latest]}, fp)
        # endwith
        list__artifact.append(path_metrics_latest)

        # print
        header = f' [ {", ".join([ds_name[:3] for ds_name in self.list__name])}]'
        mtx = np.array2string(self.nda_acc,
                              formatter={'float_kind': (lambda x: f'{x:.2f}' if x == x else ' nan')},
                              max_line_width=np.inf)
        print(f'acc:\n{header}\n{mtx}')
        print({k: v for k, v in dict__idx_task__metrics[idx_task_latest].items() if k.startswith('acc')})

        return dict__idx_task__metrics, list__artifact
    # enddef
