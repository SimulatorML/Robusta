from typing import Tuple, Any, Dict

import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator


def _plot_subset(fs: object,
                 **kwargs) -> Tuple:
    """
    Plots the score versus the number of selected features for each trial in a feature selection object.

    Args:
        fs (object): Feature selection object with a trials_ attribute containing a list of namedtuples.
        **kwargs: Additional arguments to be passed to the ax.plot() function.

    Returns:
        fig (matplotlib.figure.Figure): The figure object created.
        ax (matplotlib.axes.Axes): The axes object created.
    """
    fig, ax = plt.subplots(1, 1)

    # Set the title and labels of the plot
    ax.set_title(type(fs).__name__)
    ax.set_xlabel('# features')
    ax.set_ylabel('score')

    # Check if the feature selection was performed in a forward or backward fashion
    forward = getattr(fs, 'forward', True)

    # Invert the x-axis if the feature selection was performed in a backward fashion
    if not forward:
        plt.gca().invert_xaxis()

    # Iterate through each trial and plot the scores versus the number of selected features
    for trial in fs.trials_:

        # Current Point
        x1 = trial.n_selected
        y1 = trial.score

        ax.plot([x1], [y1], **kwargs)

        # Previous Point
        if hasattr(trial, 'parents'):

            # Iterate through each parent trial and plot the scores versus the number of selected features
            for parent in trial.parents:

                if not hasattr(parent, 'score'): continue

                x0 = parent.n_selected
                y0 = parent.score

                ax.plot([x0, x1], [y0, y1], **kwargs)

    fig.show()
    return fig, ax


def _plot_progress(fs: Any, **kwargs: Dict[str, Any]) -> tuple:
    """
    Plot the progress of a hyperparameter optimization algorithm.

    Args:
        fs (Any): A hyperparameter optimization algorithm object.
        **kwargs (Dict[str, Any]): Keyword arguments to be passed to `matplotlib`'s plot function.

    Returns:
        tuple: A tuple containing a `matplotlib` figure and axis object.
    """

    # Create a figure and axis object for plotting
    fig, ax = plt.subplots(1, 1)

    # Set the title, x-label, and y-label for the plot
    ax.set_title(type(fs).__name__)
    ax.set_xlabel('iters')
    ax.set_ylabel('score')

    # Loop over each trial in the optimization algorithm
    for i, trial in enumerate(fs.trials_):

        # Get the current point's x and y coordinates
        x1 = trial.idx
        y1 = trial.score

        # Plot the current point
        ax.plot([x1], [y1], **kwargs)

        # Check if the current trial has parents
        if hasattr(trial, 'parents'):

            # Loop over each parent of the current trial
            for parent in trial.parents:

                # Check if the parent has a score attribute
                if not hasattr(parent, 'score'):
                    continue

                # Get the parent's x and y coordinates
                x0 = parent.idx
                y0 = parent.score

                # Plot a line between the parent and current point
                ax.plot([x0, x1], [y0, y1], **kwargs)

    # Set the x-axis to display only integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Show the plot
    fig.show()

    # Return the figure and axis object
    return fig, ax
