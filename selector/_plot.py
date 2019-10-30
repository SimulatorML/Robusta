import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator

import numpy as np



def _plot_progress(fs, marker='.', color='g', alpha=0.6):

    fig, ax = plt.subplots(1,1)

    ax.set_title(type(fs).__name__)
    ax.set_xlabel('# features')
    ax.set_ylabel('score')

    forward = getattr(fs, 'forward', True)

    if not forward:
        plt.gca().invert_xaxis()

    for _, trial in fs.trials_.iterrows():

        # current point
        xx = [len(trial['subset'])]
        yy = [trial['score']]

        # previous point
        if getattr(trial, 'prev_subset', None) and np.isfinite(trial.prev_score):
            xx += [len(trial['prev_subset'])]
            yy += [trial['prev_score']]

            forward = getattr(fs, 'forward', True)
            if (xx[0] > xx[1]) == forward:
                line_style = marker + '-'
                alpha0 = alpha
            else:
                line_style = marker + '--'
                alpha0 = alpha / 2
        else:
            line_style = marker
            alpha0 = alpha

        ax.plot(xx, yy, line_style, c=color, alpha=alpha0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()
