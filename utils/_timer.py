import time




class Timer:
    def __init__(self, text=None):
        self.text = text

    def __enter__(self):
        self.cpu = time.clock()
        self.time = time.time()
        if self.text:
            logmsg("{}...".format(self.text))
        return self

    def __exit__(self, *args):
        self.cpu = time.clock() - self.cpu
        self.time = time.time() - self.time
        if self.text:
            logmsg("{}: cpu {}, time {}\n".format(self.text, secfmt(self.cpu), secfmt(self.time)))




def secfmt(s):
    H, r = divmod(s, 3600)
    M, S = divmod(r, 60)
    if H:
        return '{} h {} min {} sec'.format(int(H), int(M), int(S))
    elif M:
        return '{} min {} sec'.format(int(M), int(S))
    elif S >= 1:
        return '{} sec'.format(int(S))
    else:
        return '{} ms'.format(int(S*1000))


def logmsg(msg):
    for m in msg.split('\n'):
        t = datetime.datetime.now().strftime("[%H:%M:%S]")
        print(t, m)
        time.sleep(0.01)
