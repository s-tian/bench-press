import sys
import importlib


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


