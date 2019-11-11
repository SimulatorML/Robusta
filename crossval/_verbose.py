import pandas as pd
import numpy as np

import datetime, time
import termcolor

from queue import PriorityQueue
from robusta import utils




__all__ = ['CVLogger']




class CVLogger:

    def __init__(self, estimator, cv, verbose=1, n_digits=4, compact=False):

        self.name = estimator.__class__.__name__
        self.n_folds = cv.get_n_splits()

        self.n_digits = n_digits
        self.verbose = verbose
        self.compact = compact

        self.queue = PriorityQueue()
        self.next = 0


    def log(self, fold, result):

        if fold is -1:
            return

        # Append current fold's message
        msg = '{:.{n}f}'.format(result['score'], n=self.n_digits)
        if not self.compact:
            msg = ' FOLD{:>3}:   '.format(fold) + msg

        self.queue.put((fold, msg))

        # Print all previous
        while not self.queue.empty():

            fold, msg = self.queue.get()

            if self.next == fold:
                self._log(msg)
                self.next += 1

            else:
                self.queue.put((fold, msg))
                break


    def start(self):

        if not self.compact and self.verbose >= 2:
            utils.logmsg(' ' + self.name)
            print()


    def end(self, result):

        if not self.verbose:
            return

        m = '{:.{n}f}'.format(np.mean(result['score']), n=self.n_digits)
        s = '{:.{n}f}'.format(np.std(result['score']), n=self.n_digits)
        m = termcolor.colored(m, 'yellow')

        if self.compact:
            if self.verbose >= 2:
                self._log(f'{m} ± {s}')
                self._log(f'[{self.name}]', end='\n')

        else:
            if self.verbose > 1:
                print()

            self._log(f' AVERAGE:   {m} ± {s}')
            print()


    def _log(self, msg, end=' '*4):
        if not self.verbose:
            return
        if self.compact:
            print(msg, end=end)
        else:
            utils.logmsg(msg)
        time.sleep(0.01)
