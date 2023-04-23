import time
from queue import PriorityQueue
from typing import Dict

import numpy as np
import termcolor
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from .. import utils


class CVLogger:
    """
    A class for logging cross-validation results of a given estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object to be used for cross-validation.
    cv : BaseCrossValidator
        The cross-validation object to be used for splitting the data.
    verbose : int, optional
        The verbosity level of logging. The higher the level, the more detailed the logging.
        0: no logging
        1: print mean and standard deviation of scores for each fold and training/validation sets.
        2: print the name of the estimator class.
        Default is 1.
    n_digits : int, optional
        The number of digits to display for scores. Default is 4.
    compact : bool, optional
        Whether to print compactly or not. If True, print all the scores in a single line.
        If False, print each score in a separate line. Default is False.

    Attributes
    ----------
    name : str
        The name of the estimator class.
    n_folds : int
        The number of splits for cross-validation.
    n_digits : int
        The number of digits to display for scores.
    verbose : int
        The verbosity level of logging.
    compact : bool
        Whether to print compactly or not.
    queue : PriorityQueue
        A priority queue to store messages with their fold numbers.
    next : int
        The next fold number to print the message for.

    Methods
    -------
    log(fold: int, result: Dict[str, float]) -> None
        Log the given result for the given fold number.
    start() -> None
        Print the name of the estimator class if not in compact mode and verbosity level is at least 2.
    end(result: Dict[str, float]) -> None
        Print the mean and standard deviation of training and validation scores if available in the result.
    _log(msg: str, end: str) -> None
        Log the given message with the given end character.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 cv: BaseCrossValidator,
                 verbose: int = 1,
                 n_digits: int = 4,
                 compact: bool = False):
        # Get the name of the estimator class
        self.name = estimator.__class__.__name__

        # Get the number of splits for cross-validation
        self.n_folds = cv.get_n_splits()

        # Number of digits to display for scores
        self.n_digits = n_digits

        # Verbosity level (0, 1, or 2)
        self.verbose = verbose

        # Whether to print compactly or not
        self.compact = compact

        # A priority queue to store messages with their fold numbers
        self.queue = PriorityQueue()

        # The next fold number to print the message for
        self.next = 0

    def log(self,
            fold: int,
            result: Dict[str, float]) -> None:

        # Log the given result for the given fold number
        if fold is -1:
            # Do not log if the fold number is -1
            return

        msg = ''
        if 'trn_score' in result:
            msg0 = '{:.{n}f}'.format(result['trn_score'], n=self.n_digits)
            msg0 = f' TRN {fold + 1:>2}:   {msg0}'
            # Add the training score message to the overall message
            msg += msg0

        if 'val_score' in result:
            msg1 = '{:.{n}f}'.format(result['val_score'], n=self.n_digits)
            msg1 = f' VAL {fold + 1:>2}:   {msg1}'
            msg += ' ' * 3 if msg else ''
            # Add the validation score message to the overall message
            msg += msg1

        # Add the message to the queue with its fold number
        self.queue.put((fold, msg))
        while not self.queue.empty():

            # Get the message with the smallest fold number from the queue
            fold, msg = self.queue.get()
            if self.next == fold:

                # Print the message if the next fold number is the same as the current fold number
                self._log(msg)

                # Increment the next fold number to print the message for
                self.next += 1
            else:

                # Put the message back to the queue if the fold numbers do not match
                self.queue.put((fold, msg))
                break

    def start(self) -> None:
        # Print the name of the estimator class if not in compact mode and verbosity level is at least 2
        if not self.compact and self.verbose >= 2:
            utils.logmsg(' ' + self.name)
            print()

    def end(self,
            result: Dict[str, float]) -> None:

        # Print the mean and standard deviation of training and validation scores if available in the result
        if not self.verbose:
            # Do not print if verbosity level is 0
            return

        # Check if training score is present in the result
        if 'trn_score' in result:
            # Compute the mean and standard deviation of training scores
            mt = '{:.{n}f}'.format(np.mean(result['trn_score']), n=self.n_digits)
            st = '{:.{n}f}'.format(np.std(result['trn_score']), n=self.n_digits)
            # Color the mean of training scores blue
            mt = termcolor.colored(mt, 'blue')

        # Check if validation score is present in the result
        if 'val_score' in result:
            # Compute the mean and standard deviation of validation scores
            mv = '{:.{n}f}'.format(np.mean(result['val_score']), n=self.n_digits)
            sv = '{:.{n}f}'.format(np.std(result['val_score']), n=self.n_digits)
            # Color the mean of validation scores yellow
            mv = termcolor.colored(mv, 'yellow')

        if self.compact:
            if self.verbose >= 2:
                # Print the mean and standard deviation of training scores if available
                if 'trn_score' in result: self._log(f'{mt} ± {st}')
                # Print the mean and standard deviation of validation scores if available
                if 'val_score' in result: self._log(f'{mv} ± {sv}')
                # Print the name of the estimator class
                self._log(f'[{self.name}]', end='\n')
        else:
            if self.verbose >= 2:
                # Print a newline character
                print()

            if self.verbose >= 1:
                # Print the mean and standard deviation of training scores if available
                if 'trn_score' in result: self._log(f' TRAIN:   {mt} ± {st}')
                # Print the mean and standard deviation of validation scores if available
                if 'val_score' in result: self._log(f' VALID:   {mv} ± {sv}')
                # Print a newline character
                print()

    def _log(self,
             msg: str,
             end: str = ' ' * 4) -> None:
        # If verbosity is 0, do not print anything
        if not self.verbose:
            return
        if self.compact:
            # Print the message with the given end character
            print(msg, end=end)
        else:
            # Log the message
            utils.logmsg(msg)
        # Wait for a short time to prevent printing issues
        time.sleep(0.01)
