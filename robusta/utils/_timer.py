import datetime
import time
from typing import Optional


def logmsg(msg: str) -> None:
    """
    Prints a message with a timestamp in the format of [HH:MM:SS].

    Args:
        msg (str): A string message to be logged. Can contain newlines.

    Returns:
        None
    """
    # split the message into separate lines
    for m in msg.split('\n'):
        # get the current time and format it as [HH:MM:SS]
        t = datetime.datetime.now().strftime("[%H:%M:%S]")
        # print the timestamp and the message line
        print(t, m)
        # pause briefly to avoid overloading the output stream
        time.sleep(0.01)

def secfmt(s: float) -> str:
    """
    Formats a number of seconds as a string.

    Args:
        s (float): The number of seconds to be formatted.

    Returns:
        str: A string representing the formatted time, in the format of "H h M min S sec" or "M min S sec" or "S sec" or "X ms".
    """
    # convert seconds to hours, minutes, and seconds
    H, r = divmod(s, 3600)
    M, S = divmod(r, 60)

    # check if there are hours, minutes, or just seconds
    if H:
        # format as "H h M min S sec"
        return '{} h {} min {} sec'.format(int(H), int(M), int(S))
    elif M:
        # format as "M min S sec"
        return '{} min {} sec'.format(int(M), int(S))
    elif S >= 1:
        # format as "S sec"
        return '{} sec'.format(int(S))
    else:
        # format as "X ms"
        return '{} ms'.format(int(S * 1000))

class Timer:
    """
    A context manager that measures CPU time and wall-clock time.

    Usage:
    with Timer('some operation'):
        # code to be timed
    """

    def __init__(self,
                 text: Optional[str] = None):
        """
        Initializes a Timer object.

        Args:
            text (Optional[str]): A string describing the operation being timed.
        """
        self.text = text

    def __enter__(self) -> 'Timer':
        """
        Called when the Timer is entered.

        Returns:
            Timer: The Timer object.
        """
        # record the starting CPU time and wall-clock time
        self.cpu = time.clock()
        self.time = time.time()

        # log the start of the timed operation, if a description is provided
        if self.text:
            logmsg("{}...".format(self.text))

        # return the Timer object
        return self

    def __exit__(self,
                 *args) -> None:
        """
        Called when the Timer is exited.

        Args:
            *args: Exception information, if any.
        """
        # calculate the elapsed CPU time and wall-clock time
        self.cpu = time.clock() - self.cpu
        self.time = time.time() - self.time

        # log the results of the timed operation, if a description is provided
        if self.text:
            logmsg("{}: cpu {}, time {}\n".format(self.text, secfmt(self.cpu), secfmt(self.time)))
