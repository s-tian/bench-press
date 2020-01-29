import sys
import importlib


# Stack overflow #1176136
def str_to_class(path_name):
    pathname_split = path_name.split('.')
    module_name, class_name = '.'.join(pathname_split[:-1]), pathname_split[-1]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
