import pandas as pd
import numpy as np



def sec_to_str(s):
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
