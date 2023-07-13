#!/usr/bin/env python3
import os
import pickle
import tempfile
from datetime import datetime
from typing import *

import hydra
import mlflow
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from optuna import Trial

import approaches
import utils
from approaches.abst_appr import AbstractAppr
from dataloader import get_shuffled_dataloder
from mymetrics import MyMetrics
from utils import BColors, myprint as print, suggest_float, suggest_int


def instance_appr(trial: Trial, cfg: DictConfig,
                  list__ncls: List[int], inputsize: Tuple[int, ...]) -> AbstractAppr:
    if cfg.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.device
    # endif
    print(f'device: {device}', bcolor=BColors.OKBLUE)

    seed_pt = suggest_int(trial, cfg, 'seed_pt')
    utils.set_seed_pt(seed_pt)

    appr_args = {
        'device': device,
        'list__ncls': list__ncls,
        'inputsize': inputsize,
        'lr': cfg.lr,
        'lr_factor': cfg.lr_factor,
        'lr_min': cfg.lr_min,
        'epochs_max': cfg.epochs_max,
        'patience_max': cfg.patience_max,
        'backbone': cfg.backbone.name,
        'nhid': cfg.nhid,
        }

    def fetch_param_float(*pnames: str) -> float:
        v = suggest_float(trial, cfg, 'appr', 'tuned', cfg.seq.name, pnames[-1])

        return v
    # enddef

    if cfg.appr.name.lower() == 'spg':
        if appr_args['backbone'] in ['alexnet']:
            appr_args['drop1'] = fetch_param_float('drop1')
            appr_args['drop2'] = fetch_param_float('drop2')
        else:
            raise NotImplementedError
        # endif

        appr_args['lamb'] = 0

        appr = approaches.appr_spg.Appr(appr_args)
    else:
        raise NotImplementedError(cfg.appr.name)
    # endif

    return appr


def load_dataloader(cfg: DictConfig) -> Dict[int, Dict[str, Any]]:
    basename_data = f'seq={cfg.seq.name}_bs={cfg.seq.batch_size}_seed={cfg.seed}'
    dirpath_data = os.path.join(hydra.utils.get_original_cwd(), 'data')

    # load data
    filepath_pkl = os.path.join(dirpath_data, f'{basename_data}.pkl')
    if os.path.exists(filepath_pkl):
        with open(filepath_pkl, 'rb') as f:
            dict__idx_task__dataloader = pickle.load(f)
        # endwith

        print(f'Loaded from {filepath_pkl}', bcolor=BColors.OKBLUE)
    else:
        dict__idx_task__dataloader = get_shuffled_dataloder(cfg)
        with open(filepath_pkl, 'wb') as f:
            pickle.dump(dict__idx_task__dataloader, f)
        # endwith
    # endif

    # compute hash
    num_tasks = len(dict__idx_task__dataloader.keys())
    hash = []
    for idx_task in range(num_tasks):
        name = dict__idx_task__dataloader[idx_task]['fullname']
        ncls = dict__idx_task__dataloader[idx_task]['ncls']
        num_train = len(dict__idx_task__dataloader[idx_task]['train'].dataset)
        num_val = len(dict__idx_task__dataloader[idx_task]['val'].dataset)
        num_test = len(dict__idx_task__dataloader[idx_task]['test'].dataset)

        msg = f'idx_task: {idx_task}, name: {name}, ncls: {ncls}, num: {num_train}/{num_val}/{num_test}'
        hash.append(msg)
    # endfor
    hash = '\n'.join(hash)

    # check hash
    filepath_hash = os.path.join(dirpath_data, f'{basename_data}.txt')
    if os.path.exists(filepath_hash):
        with open(filepath_hash, 'rt') as f:
            hash_target = f.read()
        # endwith

        assert hash_target == hash

        print(f'Succesfully matched to {filepath_hash}', bcolor=BColors.OKBLUE)
        print(hash)
    else:
        # save hash
        with open(filepath_hash, 'wt') as f:
            f.write(hash)
        # endwith
    # endif

    return dict__idx_task__dataloader


def outer_objective(cfg: DictConfig, expid: str) -> Callable[[Trial], float]:
    dict__idx_task__dataloader = load_dataloader(cfg)

    num_tasks = len(dict__idx_task__dataloader.keys())
    list__name = [dict__idx_task__dataloader[idx_task]['name'] for idx_task in range(num_tasks)]
    list__ncls = [dict__idx_task__dataloader[idx_task]['ncls'] for idx_task in range(num_tasks)]
    inputsize = dict__idx_task__dataloader[0]['inputsize']  # type: Tuple[int, ...]

    def objective(trial: Trial) -> float:
        appr = instance_appr(trial, cfg, list__ncls=list__ncls, inputsize=inputsize)

        with mlflow.start_run(experiment_id=expid):
            mlflow.log_params(trial.params)
            print(f'\n'
                  f'******* trial params *******\n'
                  f'{trial.params}\n',
                  f'****************************', bcolor=BColors.OKBLUE)

            list__dl_test = [dict__idx_task__dataloader[idx_task]['test'] for idx_task in range(num_tasks)]
            mm = MyMetrics(list__name, list__dl_test=list__dl_test)

            for idx_task in range(num_tasks):
                dl_train = dict__idx_task__dataloader[idx_task]['train']
                dl_val = dict__idx_task__dataloader[idx_task]['val']

                results_train = appr.train(idx_task=idx_task, dl_train=dl_train, dl_val=dl_val)
                epoch = results_train['epoch']
                time_consumed = results_train['time_consumed']

                appr.complete_learning(idx_task=idx_task, dl_train=dl_train, dl_val=dl_val)
                mm.add_record_misc(idx_task, epoch=epoch, time_consumed=time_consumed)

                # test for all previous tasks
                for t_prev in range(idx_task + 1):
                    results_test = appr.test(t_prev, dict__idx_task__dataloader[t_prev]['test'])
                    loss_test, acc_test = results_test['loss_test'], results_test['acc_test']
                    print(f'[{t_prev}] acc: {acc_test:.3f}')

                    mm.add_record(idx_task_learned=idx_task, idx_task_tested=t_prev,
                                  loss=loss_test, acc=acc_test)
                # endfor

                # save artifacts
                with tempfile.TemporaryDirectory() as dir:
                    print(f'ordinary train/test after learning {idx_task}')
                    for mmm in [mm]:
                        idxs = []
                        metrics_final, list__artifacts = mmm.save(dir, idx_task, indices_task_ignored=idxs)
                        for k, v in metrics_final[idx_task].items():
                            trial.set_user_attr(k, v)
                        # endfor

                        mlflow.log_metrics(metrics_final[idx_task], step=idx_task)
                        for artifact in list__artifacts:
                            mlflow.log_artifact(artifact)
                        # endfor
                    # endfor
                # endwith
            # endfor

            obj = metrics_final[num_tasks - 1]['acc__Overall']
        # endwith

        print(f'Emptying CUDA cache...')
        torch.cuda.empty_cache()

        return obj
    # enddef

    return objective


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(f'\n{OmegaConf.to_yaml(cfg)}')

    utils.set_seed(cfg.seed)
    mlflow.pytorch.autolog()
    expname = cfg.expname
    expid = mlflow.create_experiment(expname)
    n_trials = cfg.n_trials

    study = optuna.create_study(direction=cfg.optuna.direction,
                                storage=cfg.optuna.storage,
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                load_if_exists=False,
                                study_name=expname,
                                )

    study.set_user_attr('Completed', False)
    study.optimize(outer_objective(cfg, expid), n_trials=n_trials,
                   gc_after_trial=True, show_progress_bar=True)
    study.set_user_attr('Completed', True)

    print(f'best params: {study.best_params}')
    print(f'best value: {study.best_value}')
    print(study.trials_dataframe())
    print(f'{expname}')


if __name__ == '__main__':
    OmegaConf.register_new_resolver('now', lambda pattern: datetime.now().strftime(pattern))
    main()
