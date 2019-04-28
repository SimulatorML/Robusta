from IPython import display
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt
import numpy as np

from robusta import utils, metrics




__all__ = ['print_progress', 'plot_progress']


def print_progress(fs):

    trial = fs.trials_[-1]

    if fs.verbose >= 1:
        # number of active features
        k = len(trial['cols'])
        n = fs.max_cols

        # scores
        score = abs(trial['score'])
        best_score = abs(fs.best_score_)
        metric_name = fs.cv.metric_name
        star = '(*)' if score == best_score else ''

        msg = (utils.ctime_str(), k, n, metric_name, score, best_score, star)
        print('%s active cols: %i/%i     %s: %.4f     best: %.4f %s' % msg)

    if fs.verbose >= 2:
        # current features subset
        print(trial['cols'])
        print()



def plot_progress(fs):

    # init if first time
    if not hasattr(fs, 'ax') or not getattr(fs, 'ax'):
        fs.fig, fs.ax = plt.subplots(1,1)

        fs.ax.set_title('Feature Selection ({})'.format(type(fs).__name__))
        fs.ax.set_xlabel('# features')
        fs.ax.set_ylabel(fs.cv.metric_name)

        if not fs.forward: plt.gca().invert_xaxis()

    # last point
    trial = fs.trials_[-1]

    xx = [trial['n_cols']]
    yy = [trial['score']]

    # previous point
    prev_trial = fs.find_trial(trial['prev'])

    if prev_trial:
        xx.insert(0, prev_trial['n_cols'])
        yy.insert(0, prev_trial['score'])

        line_style = 'o-' if (xx[0] < xx[1]) == fs.forward else 'o--'
    else:
        line_style = 'o'

    color = metrics.metric_color[fs.cv.metric_name]
    fs.ax.plot(xx, np.abs(yy), line_style, c=color, alpha=0.7)
    fs.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fs.fig.canvas.draw()
