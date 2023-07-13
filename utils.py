import datetime
import inspect
import os
import pprint
import random
import sys
from enum import Enum
from typing import *

import numpy as np
import pandas as pd
import pytz
import sklearn
import torch
from omegaconf import DictConfig
from optuna import Trial
from torch import Tensor, optim


class BColors(Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    LIGHTGRAY = '\033[90m'
    DARKGRAY = '\033[30m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def myprint(*args: Any, sep: str = ' ', begin: str = '', end: str = '\n',
            num_stacks=3, bcolor: BColors = None,
            original: bool = False, pretty: bool = False, flush: bool = True) -> None:
    """My-custom print function.

    Args:
        *args ():
        sep (str):
        begin (str):
        end (str):
        num_stacks (int):
        bcolor ():
        original (bool):
        pretty (bool):
        flush (bool):

    Returns:
        (None):
    """
    assert_type(args, object)
    assert_type(sep, str)
    assert_type(begin, str)
    assert_type(end, str)
    assert_type(num_stacks, int)
    assert_type(bcolor, BColors, allow_none=True)
    assert_type(original, bool)
    assert_type(pretty, bool)
    assert_type(flush, bool)

    tz = pytz.timezone('America/Chicago')
    dtstr = datetime2str(datetime.datetime.now(tz), fmt='%Y/%m/%d %H:%M:%S')
    try:
        stacks = reversed(inspect.stack()[1:num_stacks + 1])
        callers = ' > '.join(map(lambda fr: '{}(L{})'.format(fr[3], fr[2]), stacks))
    except IndexError:
        callers = '_error_'
    # endtry

    if original:
        msg = args
        print(*msg, sep=sep, end=end, flush=flush)
    else:
        head_info = BColors.LIGHTGRAY.value
        tail_info = BColors.ENDC.value

        if bcolor is None:
            head_body = ''
            tail_body = ''
        else:
            head_body = bcolor.value
            tail_body = BColors.ENDC.value
        # endif

        msg = f'{begin}{head_info}[{dtstr}] {callers} |{tail_info}{head_body} {" ".join(map(str, args))}{tail_body}'
        if pretty:
            pprint(msg)
        else:
            print(msg, sep=sep, end=end, flush=flush)
        # endif
    # endif
    if flush:
        sys.stdout.flush()
    # endif


def suggest_int(trial: Trial, cfg: DictConfig, *route: str) -> int:
    return _suggest(trial.suggest_int, cfg, *route)


def suggest_float(trial: Trial, cfg: DictConfig, *route: str) -> float:
    return _suggest(trial.suggest_float, cfg, *route)


def _suggest(func: Callable, cfg: DictConfig, *route: str) -> Union[float, int]:
    d = cfg
    for p in route:
        d = d[p]
    # endfor
    name = route[-1]

    if isinstance(d, DictConfig):
        low = d['low']
        high = d['high']
        if 'step' in d.keys():
            step = d['step']
            v = func(name, low, high, step=step)
        else:
            v = func(name, low, high)
        # endif
    else:
        v = d
    # endif

    print(f'Fetching name: {name}={v} from {"/".join(route)}', flush=True)
    return v


def set_seed(seed: int):
    myprint(f'Set seed={seed}', bcolor=BColors.OKBLUE)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_seed_pt(seed_pt: int):
    myprint(f'Set seed_pt={seed_pt}', bcolor=BColors.OKBLUE)
    # pytorch
    torch.manual_seed(seed_pt)
    torch.cuda.manual_seed_all(seed_pt)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def datetime2str(dt: datetime.datetime = None, fmt: str = '%Y%m%d_%H%M%S') -> str:
    """Convert datetime into str format.

    Args:
        dt (datetime.datetime or None):
        fmt (str):

    Returns:
        (str):
    """
    assert_type(dt, datetime.datetime, allow_none=True)
    assert_type(fmt, str)

    if dt is None:
        dt = datetime.datetime.now()
    # endif

    dtstr = dt.strftime(fmt)
    return dtstr


def get_current_lr(optimizer: optim.Optimizer) -> float:
    list__lr = [param_group['lr'] for param_group in optimizer.param_groups]
    assert len(list__lr) == 1, list__lr
    lr = list__lr[0]

    return lr


def my_accuracy(y_true: Union[Tensor, np.ndarray], y_pred: Union[Tensor, np.ndarray]) -> Tensor:
    assert_type(y_true, [Tensor, np.ndarray])
    assert_type(y_pred, [Tensor, np.ndarray])

    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    # endif
    n = y_true.shape[0]
    assert_shape(y_true, n)

    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    # endif
    if y_pred.ndim == 1:
        pass
    elif y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)
    else:
        raise ValueError
    # endif
    assert_shape(y_pred, n)

    acc = torch.tensor(sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred), requires_grad=False)
    assert_type(acc, Tensor)
    return acc


def assert_type(args: Any, *types: Union[type, Iterable[type]], allow_none: bool = False) -> bool:
    def candidate_types(*types: Union[type], list___series=None) -> List[List[type]]:
        if list___series is None:
            list___series = [[]]
        # endif

        t = types[0]
        if not isinstance(t, (list, tuple, dict)):
            t = [t]
        # endif

        ls = []
        for _t in t:
            if _t in [list, tuple, dict] and len(types) > 1:
                for series in candidate_types(*types[1:], list___series=list___series):
                    series.insert(0, _t)
                    ls.append(series)
                # endfor
            else:
                ls.append([_t])
            # endif
        # endfor

        return ls
    # enddef

    def check_rec(values: Any, list___types: List[List[type]]) -> bool:
        flag = (allow_none or any([isinstance(values, types[0]) for types in list___types]))

        if isinstance(values, (list, tuple, dict)):
            if isinstance(values, dict):
                values = values.values()
            # endif
            for value in values:
                flag_any = all([len(types) <= 1 for types in list___types])
                for types in list___types:
                    if len(types) > 1:
                        flag_any |= check_rec(value, [types[1:]])
                    # endif
                # endfor
                flag &= flag_any
            # endfor
        # endfor

        return flag
    # enddef

    list___types = candidate_types(*types)  # type: List[List[type]]
    flag_any = check_rec(args, list___types)

    assert flag_any, _make_assert_msg(args, types)


def assert_shape(x: Union[np.ndarray, Tensor, pd.DataFrame, pd.Series, list, set, tuple], *dims: int,
                 allow_none: bool = False) -> None:
    """Assert the argument's shape.

    Args:
        x (np.ndarray or torch.Tensor or pd.DataFrame):
        *dims (List[int]):
        allow_none (bool):

    Returns:
        (None):
    """
    assert_type(x, [np.ndarray, torch.Tensor, pd.DataFrame, pd.Series, list, set, tuple], object, allow_none=allow_none)
    assert_type(dims, tuple, int)
    assert_type(allow_none, bool)

    if allow_none and x is None:
        return
    # endif

    if isinstance(x, (list, set, tuple)):
        shape = (len(x),)
    else:
        shape = x.shape
    # endif

    assert len(shape) == len(dims), f'actual: {shape} vs expected: {dims}'
    for i in range(len(dims)):
        if dims[i] == -1:
            isinstance(shape[i], int), f'actual: {shape}[{i}] vs expected: {dims}[{i}]'
        else:
            assert shape[i] == dims[i], f'actual: {shape}[{i}] vs expected: {dims}[{i}]'
        # endif
    # endfor


# --------------------
# Private functions
# --------------------

def _make_assert_msg(actual: Any, type_expected: Any) -> str:
    """Make a message that will be shown the assertion fails.

    Args:
        actual (Any):

    Returns:
        (str): assertion message
    """
    assert_type(actual, object)
    assert_type(type_expected, object)

    return f'[actual] type: {type(actual)}, value: {actual}, [expected] type: {type_expected}'


def _candidate_types(*types: Union[type], list___series=None) -> List[List[type]]:
    if list___series is None:
        list___series = [[]]
    # endif

    t = types[0]
    if not isinstance(t, (list, tuple, dict)):
        t = [t]
    # endif

    l = []
    for _t in t:
        if _t in [list, tuple, dict] and len(types) > 1:
            for series in _candidate_types(*types[1:], list___series=list___series):
                series.insert(0, _t)
                l.append(series)
            # endfor
        else:
            l.append([_t])
        # endif
    # endfor

    return l
