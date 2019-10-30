from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt
import seaborn as sns

import numpy as np



def _plot_progress(opt, marker='.', color='#eeaa24', alpha=0.8):

    fig, ax = plt.subplots(1,1)

    ax.set_title(type(opt).__name__)
    ax.set_xlabel('iters')
    ax.set_ylabel('score')

    trials = opt.trials_

    sns.regplot(trials.index+1, 'score', trials, color=color)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.show()
