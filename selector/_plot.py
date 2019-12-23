import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator

import numpy as np



def _plot_subset(fs, **kwargs):

    fig, ax = plt.subplots(1,1)

    ax.set_title(type(fs).__name__)
    ax.set_xlabel('# features')
    ax.set_ylabel('score')

    forward = getattr(fs, 'forward', True)

    if not forward:
        plt.gca().invert_xaxis()

    for trial in fs.trials_:

        # Current Point
        x1 = trial.n_selected
        y1 = trial.score

        ax.plot([x1], [y1], **kwargs)

        # Previous Point
        if hasattr(trial, 'parents'):

            for parent in trial.parents:

                if not hasattr(parent, 'score'): continue

                x0 = parent.n_selected
                y0 = parent.score

                ax.plot([x0, x1], [y0, y1], **kwargs)

    fig.show()
    return fig, ax


def _plot_progress(fs, **kwargs):

    fig, ax = plt.subplots(1,1)

    ax.set_title(type(fs).__name__)
    ax.set_xlabel('iters')
    ax.set_ylabel('score')

    for i, trial in enumerate(fs.trials_):

        # Current Point
        x1 = trial.idx
        y1 = trial.score

        ax.plot([x1], [y1], **kwargs)

        # Previous Point
        if hasattr(trial, 'parents'):

            for parent in trial.parents:

                if not hasattr(parent, 'score'): continue

                x0 = parent.idx
                y0 = parent.score

                ax.plot([x0, x1], [y0, y1], **kwargs)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.show()
    
    return fig, ax
