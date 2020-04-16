# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['split_array', 'r3', 'Aggs', 'mse', 'rmse', 'export_from_data_to_core', 'exports_from_data_to_core']

# Cell
from .imports import *

# Cell
def split_array(data, idx): return data[:idx], data[idx:]

# Cell
def r3(x):
    "return a scalar value rounded to 3dp or the values of iterable, rounded to 3dp in a new array/tuple "
    try: return round(x, 3)
    except: pass
    res = [round(v, 3) for v in x]
    return tuple(res) if isinstance(x, tuple) else res

# Internal Cell
def agg_var(c, s, s2):
    "return the variance of `c`, `s` and `s2` - i.e. one side of `y`"
    return max((s2/c) - (s/c)**2, 0)
def agg_std(c, s, s2):
    "return the standard deviation of `c`, `s` and `s2`"
    return np.sqrt(agg_var(c, s, s2))
def agg_score(c, s, s2):
    "return the score of `c`, `s` and `s2`"
    return 0 if c == 1 else agg_std(c, s, s2)*c

# Cell
class Aggs():
    "keeps track of `c`, `s` and `s2` and provides access to `score`"
    def __init__(self, y):
        "create a new `Aggs` assuming you're going to iterate over `y`"
        self.c, self.s, self.s2 = (0, 0., 0.) # these will get updated
        self._c, self._s, self._s2 = len(y), y.sum(), (y**2).sum() # initial values are fixed
    def upd(self, yi):
        "update `c`, `s` and `s2` values with the next `y` value"
        self.c += 1
        self.s += yi
        self.s2 += yi**2
    def score(self):
        "return the sum of the standard deviation for both sides of `y`"
        c, s, s2 = self.c, self.s, self.s2
        _c, _s, _s2 = self._c, self._s, self._s2
        return agg_score(c, s, s2) + agg_score(_c-c, _s-s, _s2-s2)

# Cell
def mse(x,y): return ((x-y)**2).mean()
def rmse(x,y): return np.sqrt(mse(x, y))

# Comes from 40_test_export.ipynb, cell
def export_from_data_to_core():
    "export to a different module"
    pass

# Comes from 40_test_export.ipynb, cell
def exports_from_data_to_core():
    "exports to a different module"
    pass

# Comes from 40_test_export.ipynb, cell
def exporti_from_data_to_core():
    "internal export to a different module"
    pass