from pylab import *
import pandas as pd
import numpy as np


'''Date/Time Staff'''
import datetime, time

DT_FMT_FULL = "%Y-%m-%d %H:%M:%S"
DT_FMT_DATE = "%Y-%m-%d"
DT_FMT_TIME = '%H:%M:%S'

dt_str = lambda dt: dt.strftime(DT_FMT_FULL)
dt_dstr = lambda dt: dt.strftime(DT_FMT_DATE)
dt_tstr = lambda dt: dt.strftime(DT_FMT_TIME)
str_dt = lambda s: datetime.datetime.strptime(s, DT_FMT_FULL)
dt_flt = lambda dt: dt.timestamp()
flt_dt = lambda ts: datetime.datetime.fromtimestamp(ts)
flt_str = lambda ts: dt_str(flt_dt(ts))
flt_dstr = lambda ts: dt_dstr(flt_dt(ts))
flt_tstr = lambda ts: dt_tstr(flt_dt(ts))

ctime = lambda: datetime.datetime.now()
ctime_str = lambda: '[%s]' % dt_tstr(ctime())

def sec_to_str(s):
    H, r = divmod(s, 3600)
    M, S = divmod(r, 60)
    if H:
        return '{} h {} min {} sec'.format(int(H), int(M), int(S))
    elif M:
        return '{} min {} sec'.format(int(M), int(S))
    elif S >= 1:
        return '{} sec'.format(int(S))
    else:
        return '{} ms'.format(int(S*1000))

def td_to_str(td):
    sec = td.total_seconds()
    return sec_to_str(sec)


'''Timer Staff'''
import multiprocessing
import traceback, sys

dt = 0.01

def run_with_timeout(func, kwargs={}, timeout=None):

    def wrapper_func(queue, **params):
        try:
            result = func(**params)
            queue.put(result)

        except KeyboardInterrupt:
            raise KeyboardInterrupt

        except:
            traceback.print_exc(limit=0, file=sys.stdout)

        finally:
            queue.close()

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper_func, args=(queue,), kwargs=kwargs)
    process.start()

    try:
        if timeout:
            try:
                while timeout > 0 and queue.empty():
                    process.join(dt)
                    timeout -= dt

                if not queue.empty():
                    return queue.get_nowait()

                elif process.exitcode is None:
                    raise multiprocessing.TimeoutError

                else:
                    raise

            except:
                raise

        else:
            try:
                while process.is_alive() and queue.empty() and (not timeout or t < timeout):
                    process.join(0.01)

                if not queue.empty():
                    return queue.get_nowait()

                else:
                    raise

            except:
                raise

    except:
        raise

    finally:
        #print(process.exitcode)
        process.terminate()





'''Columns Staff'''
def cols_idx(cols, cols_sub):
    return [i for i, col in enumerate(cols) if col in cols_sub]

def cols_intersection(a, b):
    return list(set(a) & set(b))


col_dif = lambda A, B: list(A.columns.difference(B.columns))
col_com = lambda A, B: list(A.columns.intersection(B.columns))




'''Selector Staff'''
fact = lambda x: x*fact(x-1) if x else 1

def nCk(n, k):
    return fact(n)//fact(k)//fact(n-k)

def nCk_range(k_min, k_max, n):
    k_range = range(k_min, k_max+1)

    C = [nCk(n, k) for k in k_range]
    return pd.Series(C, index=k_range).sort_index()

def weighted_choice(weights):
    # weights ~ pd.Series
    rnd = np.random.random()*sum(weights)
    for i, w in weights.items():
        rnd -= w
        if rnd <= 0:
            return i


'''Integer Staff'''
identity_func = lambda x: x

def try_int(x):
    if isinstance(x, bool) == False:
        try:
            x = int(x) if int(x) == x else x
        except:
            pass
    return x

int_dict = lambda d: {key: try_int(val) for key, val in d.items()}
ind_list = lambda a: [try_int(x) for x in a]

def round_step(x, q=1, a=0, b=None):
    # x : value
    # q : step
    # a,b : left & right bounds

    x = a + ((x-a)//q)*q
    x = round(x, 8)

    if int(x) == x:
        x = int(x)
    return x



'''Parameters Staff'''
def get_params(func): return inspect.getfullargspec(func).args
def has_param(func, param): return param in get_params(func)

def extend_list(l, N): return l[:N] + [None]*(N-len(l))

def flat_dict(d):
    _d = {}
    for key, val in d.items():
        if isinstance(val, dict):
            _d.update(flat_dict(val))
        else:
            _d[key] = val
    return _d



'''Text Staff'''
import re

def get_words(s):
    return re.findall(r'\w+', s)

def count_words(s):
    return len(get_words(s))

def count_unique_words(s):
    return len(set(get_words(s)))



'''Sort Staff'''
def select_k_max(series, k):
    return series.sort_values()[::-1].dropna().iloc[:k]

def select_k_min(series, k):
    return series.sort_values().dropna().iloc[:k]


'''Plot Staff'''
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.pylab as plt


def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im
