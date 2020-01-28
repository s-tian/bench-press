import sys


# Stack overflow #1176136
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
