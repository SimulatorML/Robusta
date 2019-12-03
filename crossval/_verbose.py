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
        msg = ''

        if 'trn_score' in result:
            msg0 = '{:.{n}f}'.format(result['trn_score'], n=self.n_digits)
            msg0 = f' TRN {fold}:   {msg0}'
            msg += msg0

        if 'val_score' in result:
            msg1 = '{:.{n}f}'.format(result['val_score'], n=self.n_digits)
            msg1 = f' VAL {fold}:   {msg1}'
            msg += ' '*3 if msg else ''
            msg += msg1

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

        if 'trn_score' in result:
            mt = '{:.{n}f}'.format(np.mean(result['trn_score']), n=self.n_digits)
            st = '{:.{n}f}'.format(np.std( result['trn_score']), n=self.n_digits)
            mt = termcolor.colored(mt, 'blue')

        if 'val_score' in result:
            mv = '{:.{n}f}'.format(np.mean(result['val_score']), n=self.n_digits)
            sv = '{:.{n}f}'.format(np.std( result['val_score']), n=self.n_digits)
            mv = termcolor.colored(mv, 'yellow')

        if self.compact:
            if self.verbose >= 2:
                if 'trn_score' in result: self._log(f'{mt} ± {st}')
                if 'val_score' in result: self._log(f'{mv} ± {sv}')
                self._log(f'[{self.name}]', end='\n')

        else:
            if self.verbose >= 2: print()
            if self.verbose >= 1:
                if 'trn_score' in result: self._log(f' TRAIN:   {mt} ± {st}')
                if 'val_score' in result: self._log(f' VALID:   {mv} ± {sv}')
                print()


    def _log(self, msg, end=' '*4):
        if not self.verbose:
            return
        if self.compact:
            print(msg, end=end)
        else:
            utils.logmsg(msg)
        time.sleep(0.01)
