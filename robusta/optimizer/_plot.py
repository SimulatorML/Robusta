import matplotlib.pylab as plt
import optuna
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def _plot_progress(opt: optuna.Trial, color: str = "#eeaa24") -> None:
    """
    Plot the optimization progress of an Optuna study.

    Parameters
    ----------
    opt : BaseStudy
        The Optuna study object to plot.
    color : str, optional
        The color of the regression line in the plot. Default is '#eeaa24'.

    Returns
    -------
    Nothing:
        None
    """

    # Create a new figure with one subplot
    fig, ax = plt.subplots(1, 1)

    # Set the title and labels for the plot
    ax.set_title(type(opt).__name__)
    ax.set_xlabel("iters")
    ax.set_ylabel("score")

    # Get the trials from the Optuna study object
    trials = opt.trials_

    # Plot a regression line of the score over the index of each trial
    sns.regplot(trials.index + 1, "score", trials, color=color)

    # Set the x-axis tick locator to only show integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Display the plot
    fig.show()
