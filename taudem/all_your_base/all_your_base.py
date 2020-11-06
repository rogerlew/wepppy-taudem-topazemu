# Copyright (c) 2016-2020, University of Idaho
# All rights reserved.
#
# Roger Lew (rogerlew@gmail.com)
#
# The project described was supported by NSF award number IIA-1301792
# from the NSF Idaho EPSCoR Program and by the National Science Foundation.

from typing import Tuple

import sys
import collections
import os
from os.path import exists as _exists
from os.path import split as _split
from os.path import join as _join
from operator import itemgetter
from itertools import groupby
import shutil
from collections import namedtuple
import math

import multiprocessing

try:
    import win32com.shell.shell as shell
except:
    pass

try:
    NCPU = int(os.environ['WEPPPY_NCPU'])
except KeyError:
    NCPU = math.floor(multiprocessing.cpu_count() * 0.5)
    if NCPU < 1:
        NCPU = 1

geodata_dir = '/geodata/'

RGBA = namedtuple('RGBA', list('RGBA'))
RGBA.tohex = lambda this: '#' + ''.join('{:02X}'.format(a) for a in this)


SCRATCH = '/media/ramdisk'

if not _exists(SCRATCH):
    SCRATCH = '/Users/roger/Downloads'

if not _exists(SCRATCH):
    SCRATCH = '/workdir'

IS_WINDOWS = os.name == 'nt'


def make_symlink(src, dst):
    if IS_WINDOWS:
        if _exists(dst):
            os.remove(dst)
        params = ' '.join(['mklink', dst, src])
        shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=params)
    else:
        os.symlink(src, dst)


def cmyk_to_rgb(c, m, y, k):
    """
    """
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return r, g, b


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def find_ranges(iterable, as_str=False):
    """Yield range of consecutive numbers."""

    def func(args):
        index, item = args
        return index - item

    ranges = []
    for key, group in groupby(enumerate(iterable), func):
        group = list(map(itemgetter(1), group))
        if len(group) > 1:
            ranges.append((group[0], group[-1]))
        else:
            ranges.append(group[0])

    if not as_str:
        return ranges

    s = []

    for arg in ranges:
        if isint(arg):
            s.append(str(arg))
        else:
            s.append('{}-{}'.format(*arg))

    return ', '.join(s)


def clamp(x: float, minimum: float, maximum: float) -> float:
    x = float(x)
    if x < minimum:
        return minimum
    elif x > maximum:
        return maximum
    return x


def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    return x


def cp_chmod(src, dst, mode):
    """
    helper function to copy a file and set chmod
    """
    shutil.copyfile(src, dst)
    os.chmod(dst, mode)


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:   # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def isint(x):
    # noinspection PyBroadException
    try:
        return float(int(x)) == float(x)
    except Exception:
        return False


def isfloat(f):
    # noinspection PyBroadException
    try:
        float(f)
        return True
    except Exception:
        return False


def isbool(x):
    # noinspection PyBroadException
    return x in (0, 1, True, False)


def isnan(f):
    if not isfloat(f):
        return False
    return math.isnan(float(f))


def isinf(f):
    if not isfloat(f):
        return False
    return math.isinf(float(f))


def try_parse(f):
    # noinspection PyBroadException
    try:
        ff = float(f)
        # noinspection PyBroadException
        try:
            fi = int(f)
            return fi
        except Exception:
            return ff
    except Exception:
        return f


def try_parse_float(f):
    # noinspection PyBroadException
    try:
        return float(f)
    except Exception:
        return 0.0


def parse_name(colname):
    units = parse_units(colname)
    if units is None:
        return colname

    return colname.replace('({})'.format(units), '').strip()


def parse_units(colname):
    try:
        colsplit = colname.strip().split()
        if len(colsplit) < 2:
            return None

        if '(' in colsplit[-1]:
            return colsplit[-1].replace('(', '').replace(')', '')

        return None
    except IndexError:
        return None


class RowData:
    def __init__(self, row):

        self.row = row

    def __getitem__(self, item):
        for colname in self.row:
            if colname.startswith(item):
                return self.row[colname]

        raise KeyError

    def __iter__(self):
        for colname in self.row:
            value = self.row[colname]
            units = parse_units(colname)
            yield value, units


def c_to_f(x):
    return 9.0/5.0 * x + 32.0


def f_to_c(x):
    return (x - 32.0) * 5.0 / 9.0
