# AUTOGENERATED! DO NOT EDIT! File to edit: 73_in_ipython.ipynb (unless otherwise specified).

__all__ = ['in_ipython', 'in_ipython2']

# Cell
# current implementation copied from imports.py
import os
def in_ipython():
    "Check if the code is running in the ipython environment (jupyter including)"
    program_name = os.path.basename(os.getenv('_', ''))
    if ('jupyter-notebook' in program_name or # jupyter-notebook
        'ipython'          in program_name or # ipython
        'JPY_PARENT_PID'   in os.environ):    # ipython-notebook
        return True
    else:
        return False

# Cell
print('\nas-is', in_ipython())

# Cell
# it might be better to use get_ipython or __IPYTHON__
# https://ipython.org/ipython-doc/rel-0.10.2/html/interactive/reference.html
def in_ipython2():
    try: get_ipython(); return True
    except: return False

# Cell
print('to-be', in_ipython2())

# Cell
from nbdev.imports import IN_NOTEBOOK
print('Am I in a notebook?', IN_NOTEBOOK)