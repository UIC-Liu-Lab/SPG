import itertools
import json
import os
import pickle
import random
from typing import *

import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils import myprint as print


def get_shuffled_dataloder(cfg: DictConfig) -> Dict[int, Dict[str, Any]]:
    r = 1

    dict__idx_task__dataloader = _get_dataloaders(cfg)
    for _ in range(r):
        dict__idx_task__dataloader = _shuffle_order(dict__idx_task__dataloader)
    # endfor

    # check print
    num_tasks = len(dict__idx_task__dataloader.keys())
    for idx_task in range(num_tasks):
        for tvt in ['train']:
            name = dict__idx_task__dataloader[idx_task]['fullname']
            ncls = dict__idx_task__dataloader[idx_task]['ncls']
            dl = dict__idx_task__dataloader[idx_task][tvt]
            for x, y in dl:
                msg = f'idx_task: {idx_task:2d}({tvt}), name: {name:>17}, ncls: {ncls:2d}, num: {len(dl.dataset):>6d}, y: {y[:10]}'
                print(msg)
                break
            # endfor
        # endfor
    # endfor

    # assert
    assert all([dict__idx_task__dataloader[idx_task]['inputsize'] == dict__idx_task__dataloader[0]['inputsize']
                for idx_task in range(num_tasks)])

    return dict__idx_task__dataloader


def _get_dataloaders(cfg: DictConfig) -> Dict[int, Dict[str, Any]]:
    DIR_CWD = hydra.utils.get_original_cwd()
    DIR_DATA = os.path.join(DIR_CWD, 'data')

    mean_3ch = [0.485, 0.456, 0.406]
    std_3ch = [0.229, 0.224, 0.225]
    mean_1ch = [0.1307]
    std_1ch = [0.3081]

    def tf(inputsize: Tuple[int, int, int], mean: List[float], std: List[float]) -> transforms.Compose:
        assert len(inputsize) == 3

        return transforms.Compose([
            transforms.Resize((inputsize[1], inputsize[2]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
    # enddef

    batch_size = cfg.seq.batch_size

    num_tasks_offset = 0
    dict__idx_task__dataloader = {}
    for dataset_name in cfg.seq.dataset:
        assert dataset_name in cfg.seq.name, f'{dataset_name} NOT in {cfg.seq.name}'

        ds = cfg.seq.dataset[dataset_name]
        inputsize = ds.inputsize
        num_tasks = ds.num_tasks
        num_classes = ds.num_classes

        if dataset_name in ['cifar100_10', 'cifar100_20']:
            trf = tf(inputsize, mean_3ch, std_3ch)
            dataset_trainval = datasets.CIFAR100(root=DIR_DATA, train=True, transform=trf, download=True)
            dataset_test = datasets.CIFAR100(root=DIR_DATA, train=False, transform=trf, download=True)
        elif dataset_name in ['tinyimagenet_10', 'tinyimagenet_20']:
            trf = tf(inputsize, mean_3ch, std_3ch)
            dataset_trainval = datasets.ImageFolder(root=os.path.join(DIR_DATA, 'tiny-imagenet-200', 'train'), transform=trf)
            dataset_test = datasets.ImageFolder(root=os.path.join(DIR_DATA, 'tiny-imagenet-200', 'test'), transform=trf)
        elif dataset_name in ['imagenet_100']:
            trf = tf(inputsize, mean_3ch, std_3ch)
            dataset_trainval = Imagenet(root=os.path.join(DIR_DATA, 'imagenet'), train=True, transforms=trf)
            dataset_test = Imagenet(root=os.path.join(DIR_DATA, 'imagenet'), train=False, transforms=trf)
        elif dataset_name in ['fceleba_10', 'fceleba_20']:
            num_train_per_class = ds.num_train_per_class
            num_val_per_class = ds.num_val_per_class
            num_test_per_class = ds.num_test_per_class

            trf = tf(inputsize, mean_3ch, std_3ch)
            dataset_trainval = FCelebA(root=DIR_DATA, train=True,
                                       num_tasks=num_tasks,
                                       nsamples_per_task=(num_train_per_class + num_val_per_class) * num_classes,
                                       transforms=trf,
                                       )
            dataset_test = FCelebA(root=DIR_DATA, train=False,
                                   num_tasks=num_tasks,
                                   nsamples_per_task=num_test_per_class * num_classes,
                                   transforms=trf,
                                   )
        elif dataset_name in ['femnist_10', 'femnist_20']:
            num_train_per_class = ds.num_train_per_class
            num_val_per_class = ds.num_val_per_class
            num_test_per_class = ds.num_test_per_class

            trf = tf(inputsize, mean_1ch, std_1ch)
            dataset_trainval = FEMNIST(root=DIR_DATA, train=True,
                                       num_tasks=num_tasks,
                                       nsamples_per_task=(num_train_per_class + num_val_per_class) * num_classes,
                                       transforms=trf,
                                       )
            dataset_test = FEMNIST(root=DIR_DATA, train=False,
                                   num_tasks=num_tasks,
                                   nsamples_per_task=num_test_per_class * num_classes,
                                   transforms=trf,
                                   )
        else:
            raise NotImplementedError(dataset_name)
        # endif

        if dataset_name in ['cifar100_10', 'cifar100_20', 'tinyimagenet_10', 'tinyimagenet_20', 'imagenet_100',
                            'fceleba_10', 'fceleba_20', 'femnist_10', 'femnsit_20']:
            num_train_per_class = ds.num_train_per_class
            num_val_per_class = ds.num_val_per_class
            num_test_per_class = ds.num_test_per_class
            federated = ds.federated

            if federated:
                list__num_classes = [num_classes] * num_tasks
            else:
                list__num_classes = [len(l) for l in np.array_split(range(num_classes), num_tasks)]
            # endif
            print(f'[{dataset_name}] list__num_classes: {list__num_classes}')

            if federated:
                dict__idx_task__dataset = {idx_task: {'trainval': {'x': [], 'y': []},
                                                      'test': {'x': [], 'y': []}}
                                           for idx_task in range(num_tasks)}
                for tvt, dataset in [('trainval', dataset_trainval), ('test', dataset_test)]:
                    for t, x, y in dataset:
                        dict__idx_task__dataset[t][tvt]['x'].append(x)
                        dict__idx_task__dataset[t][tvt]['y'].append(y)
                    # endfor

                    for idx_task in range(num_tasks):
                        dict__idx_task__dataset[idx_task][tvt] = TensorDataset(
                            torch.stack(dict__idx_task__dataset[idx_task][tvt]['x'], dim=0),
                            torch.tensor(dict__idx_task__dataset[idx_task][tvt]['y']))
                    # endfor
                # endfor
            else:
                classes = random.sample(range(num_classes), k=num_classes)
                dict__idx_task__classes = {idx_task: classes[sum(list__num_classes[:idx_task])
                                                             :sum(list__num_classes[:idx_task + 1])]
                                           for idx_task in range(num_tasks)}  # type: Dict[int, List[int]]
                classes = list(itertools.chain.from_iterable(list(dict__idx_task__classes.values())))
                assert len(set(classes)) == len(classes) == num_classes

                dict__idx_task__dataset = {idx_task: {'trainval': {'x': [], 'y': []},
                                                      'test': {'x': [], 'y': []}}
                                           for idx_task in range(num_tasks)}
                for tvt, dataset in [('trainval', dataset_trainval), ('test', dataset_test)]:
                    for image, y in dataset:
                        idx_task = [t for t in range(num_tasks) if y in dict__idx_task__classes[t]]
                        assert len(idx_task) == 1
                        idx_task = idx_task[0]
                        label = dict__idx_task__classes[idx_task].index(y)
                        assert label != -1

                        dict__idx_task__dataset[idx_task][tvt]['x'].append(image)
                        dict__idx_task__dataset[idx_task][tvt]['y'].append(label)
                    # endfor

                    for idx_task in range(num_tasks):
                        # indicated size
                        if tvt == 'trainval':
                            size = (num_train_per_class + num_val_per_class) * list__num_classes[idx_task]
                        elif tvt == 'test':
                            size = num_test_per_class * list__num_classes[idx_task]
                        else:
                            raise ValueError
                        # endif

                        # reduce only if need
                        x = dict__idx_task__dataset[idx_task][tvt]['x']
                        y = dict__idx_task__dataset[idx_task][tvt]['y']
                        if len(x) == len(y) == size:
                            pass
                        else:
                            x, _, y, _ = train_test_split(x, y, stratify=y, train_size=size)
                        # endif

                        dict__idx_task__dataset[idx_task][tvt] = TensorDataset(torch.stack(x, dim=0), torch.tensor(y))
                    # endfor
                # endfor
            # endif

            # trainval -> train/val
            for idx_task in range(num_tasks):
                dataset_trainval = dict__idx_task__dataset[idx_task]['trainval']
                dataset_train, dataset_val \
                    = torch.utils.data.random_split(dataset_trainval,
                                                    lengths=[num_train_per_class * list__num_classes[idx_task],
                                                             num_val_per_class * list__num_classes[idx_task]])

                dict__idx_task__dataset[idx_task]['train'] = dataset_train
                dict__idx_task__dataset[idx_task]['val'] = dataset_val
            # endfor
        else:
            raise NotImplementedError(dataset_name)
        # endif

        # dataset -> dataloader
        for idx_task in range(num_tasks):
            dl_train = DataLoader(dict__idx_task__dataset[idx_task]['train'], batch_size=batch_size)
            dl_val = DataLoader(dict__idx_task__dataset[idx_task]['val'], batch_size=batch_size)
            dl_test = DataLoader(dict__idx_task__dataset[idx_task]['test'], batch_size=batch_size)

            assert len(dl_train.dataset) == num_train_per_class * list__num_classes[idx_task]
            assert len(dl_val.dataset) == num_val_per_class * list__num_classes[idx_task]
            assert len(dl_test.dataset) == num_test_per_class * list__num_classes[idx_task]

            dict__idx_task__dataloader[idx_task + num_tasks_offset] = {
                'name': dataset_name,
                'fullname': f'{dataset_name}-{idx_task + 1:02d}/{num_tasks}',
                'inputsize': inputsize,
                'ncls': list__num_classes[idx_task],
                'train': dl_train,
                'val': dl_val,
                'test': dl_test,
                }
        # endfor

        num_tasks_offset += num_tasks
    # endfor

    return dict__idx_task__dataloader


def _shuffle_order(dict__idx_task__dataloader: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    num_tasks = len(dict__idx_task__dataloader.keys())

    # shuffle
    indices_tasks = list(range(num_tasks))
    random.shuffle(indices_tasks)
    ret = {}
    for i, idx_task in enumerate(indices_tasks):
        ret[i] = dict__idx_task__dataloader[idx_task]
    # endfor

    return ret


class Imagenet(Dataset):
    def __init__(self, root: str, train: bool,
                 transforms: transforms.Compose):
        if train:
            data = {
                'labels': [],
                }
            for i in range(10):
                filepath = os.path.join(root, 'train', f'train_data_batch_{i + 1}')
                data_batch = self.unpickle(filepath)
                x = data_batch['data']
                y = data_batch['labels']

                if 'data' not in data.keys():
                    data['data'] = x
                else:
                    data['data'] = np.concatenate([data['data'], x], axis=0)
                # endif
                data['labels'] += y
            # endfor
        else:
            filepath = os.path.join(root, 'test', 'val_data')
            data = self.unpickle(filepath)
        # endif

        img_size = 32

        x = data['data']
        x = x.reshape((x.shape[0], img_size, img_size, 3))
        self.x = x

        self.y = data['labels']
        self.transforms = transforms

        # print(f'x.shape: {self.x.shape}')
        # print(f'y range: {np.min(self.y)}, {np.max(self.y)}')
    # enddef

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        # endwith

        return dict
    # enddef

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        x = self.x[idx]  # type: np.ndarray
        x = Image.fromarray(x.astype('uint8'), 'RGB')
        image = self.transforms(x)

        label = self.y[idx] - 1

        return image, label
    # enddef

    def __len__(self) -> int:
        return len(self.y)
    # enddef


class FCelebA(Dataset):
    def __init__(self, root: str, num_tasks: int, train: bool,
                 nsamples_per_task: int, transforms: transforms.Compose):
        super(FCelebA, self).__init__()
        self.num_tasks = num_tasks
        self.train = train
        self.tot = 'train' if self.train else 'test'
        self.num_samples = nsamples_per_task
        self.transforms = transforms

        # create dirs
        self.dir_fceleba = os.path.join(root, 'fceleba')
        self.dir_data_task = os.path.join(self.dir_fceleba, str(num_tasks))
        os.makedirs(self.dir_data_task, exist_ok=True)

        # buffer
        self.list__data = []

        # load
        self.load()
    # enddef

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, int]:
        data = self.list__data[idx]

        idx_task = data['idx_task']
        x = Image.open(os.path.join(self.dir_fceleba, 'raw', 'img_align_celeba', data['filename']))
        y = data['y']

        return idx_task, self.transforms(x), y
    # enddef

    def __len__(self) -> int:
        return len(self.list__data)
    # enddef

    def load(self):
        raw_need = False
        for idx_task in range(self.num_tasks):
            filepath_json = os.path.join(self.dir_data_task, f'{idx_task}_{self.tot}_{self.num_samples}.json')
            if not os.path.exists(filepath_json):
                raw_need = True
                break
            # endif

            with open(filepath_json, 'r') as fp:
                data = json.load(fp)
                if len(data) != self.num_samples:
                    print(f'invalid samples')
                    raw_need = True
                    break
                # endif

                self.list__data += data
            # endwith

            if raw_need:
                break
            # endif
        # endfor

        # read raw data
        if raw_need:
            self._read_raw()
            self.load()
        # endif
    # enddef

    def _read_raw(self):
        filepath_json = os.path.join(self.dir_fceleba, 'iid', self.tot, f'all_data_iid_01_0_keep_5_{self.tot}_9.json')
        with open(filepath_json, 'rt') as f:
            jdict = json.load(f)
        # endwith

        uid = 0
        for idx_user, username in enumerate(jdict['users']):
            if uid >= self.num_tasks:
                break
            # endif

            list__data = []
            for idx_data, (filename_jpg, y) in enumerate(zip(jdict['user_data'][username]['x'],
                                                             jdict['user_data'][username]['y'])):
                list__data.append({'idx_task': idx_user,
                                   'filename': filename_jpg,
                                   'y': int(y)})
            # endfor
            assert len(list__data) >= self.num_samples
            list__data = random.sample(list__data, self.num_samples)

            with open(os.path.join(self.dir_data_task, f'{idx_user}_{self.tot}_{self.num_samples}.json'), 'w') as fp:
                json.dump(list__data, fp)
            # endwith

            uid += 1
        # endfor
    # enddef


class FEMNIST(Dataset):
    def __init__(self, root: str, num_tasks: int, train: bool,
                 nsamples_per_task: int, transforms: transforms.Compose):
        super(FEMNIST, self).__init__()
        self.num_tasks = num_tasks
        self.train = train
        self.tot = 'train' if self.train else 'test'
        self.num_samples = nsamples_per_task
        self.transforms = transforms

        # create dirs
        self.dir_fceleba = os.path.join(root, 'femnist')
        self.dir_data_task = os.path.join(self.dir_fceleba, str(num_tasks))
        os.makedirs(self.dir_data_task, exist_ok=True)

        # buffer
        self.list__data = []

        # load
        self.load()
    # enddef

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, int]:
        data = self.list__data[idx]

        idx_task = data['idx_task']
        x = transforms.functional.to_pil_image(torch.tensor(data['x']).resize(1, 28, 28))
        y = data['y']

        return idx_task, self.transforms(x), y
    # enddef

    def __len__(self) -> int:
        return len(self.list__data)
    # enddef

    def load(self):
        raw_need = False
        for idx_task in range(self.num_tasks):
            filepath_json = os.path.join(self.dir_data_task, f'{idx_task}_{self.tot}_{self.num_samples}.json')
            if not os.path.exists(filepath_json):
                raw_need = True
                break
            # endif

            with open(filepath_json, 'r') as fp:
                data = json.load(fp)
                if len(data) != self.num_samples:
                    print(f'invalid samples')
                    raw_need = True
                    break
                # endif

                self.list__data += data
            # endwith

            if raw_need:
                break
            # endif
        # endfor

        # read raw data
        if raw_need:
            self._read_raw()
            self.load()
        # endif
    # enddef

    def _read_raw(self):
        for idx_task in range(self.num_tasks):
            filepath_json = os.path.join(self.dir_fceleba, 'raw', self.tot,
                                         f'all_data_{0}_iid_01_0_keep_0_{self.tot}_9.json')
            with open(filepath_json, 'rt') as f:
                jdict = json.load(f)
            # endwith

            user = jdict['users']
            assert len(user) == 1
            user = user[0]

            ns = jdict['num_samples'][0]
            user_data = jdict['user_data'][user]
            list__x = user_data['x']
            list__y = user_data['y']
            assert ns >= self.num_samples
            assert len(list__x) == len(list__y) == ns

            list__data = []
            for x, y in zip(list__x, list__y):
                list__data.append({'idx_task': idx_task,
                                   'x': x,
                                   'y': y})
            # endfor
            assert len(list__data) >= self.num_samples
            list__data = random.sample(list__data, self.num_samples)

            with open(os.path.join(self.dir_data_task, f'{idx_task}_{self.tot}_{self.num_samples}.json'), 'w') as fp:
                json.dump(list__data, fp)
            # endwith
        # endfor
    # enddef
