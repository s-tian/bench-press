import importlib

import numpy as np


# Stack overflow #1176136
def str_to_class(path_name):
    pathname_split = path_name.split('.')
    module_name, class_name = '.'.join(pathname_split[:-1]), pathname_split[-1]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def deep_map(fn, x):
    if isinstance(x, dict):
        return {x: deep_map(fn, k) for x, k in x.items()}
    if isinstance(x, list):
        return [deep_map(fn, i) for i in x]
    return fn(x)


def deep_zero_like(x):
    return deep_map(np.zeros_like, x)


def deep_ones_like(x):
    return deep_map(np.ones_like, x)


def deep_binary_apply(fn, x, y):
    """
    Requires that x and y have the same structure.
    :param fn:
    :param x:
    :param y:
    :return:
    """
    if isinstance(x, dict):
        return {key: deep_binary_apply(fn, x[key], y[key]) for key, value in x.items()}
    if isinstance(x, list):
        return [deep_binary_apply(fn, x[i], y[i]) for i in range(len(x))]
    return fn(x, y)


def deep_sum(x, y):
    return deep_binary_apply(lambda a, b: a + b, x, y)
