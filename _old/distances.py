from math import sin, cos, sqrt, atan2, radians
import pandas as pd

'''
Coordinates distances

Parameters
----------
x : tuple, Latitude & Longitude column names of the 1st coordinates
y : tuple, Latitude & Longitude column names of the 2nd coordinates

Result
----------
result : float
    Distance between two coordinates
'''
def dist_euc(x, y):
    # Euclidean distance
    dlon, dlat = _dlon_dlat(x, y)
    return sqrt(dlon**2 + dlat**2)


def dist_mnh(x, y):
    # Manhattan distance
    dlon, dlat = _dlon_dlat(x, y)
    return abs(dlon) + abs(dlat)


def dist_rad(x, y):
    # Radian distance
    dlon, dlat = _dlon_dlat(x, y)
    a = sin(dlat/2)**2 + cos(lat_x) * cos(lat_y) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6373.0 # Earth's radius
    return R * c


def _dlon_dlat(x, y):
    lat_x, long_x, lat_y, long_y = map(radians, [*x,*y])
    dlon, dlat = long_y-long_x, lat_y-lat_x
    return dlon, dlat


'''
File distances

Parameters
----------
x :
y :

Result
------
result : float
    Distance between two files
'''
def NCD(x, y, Z, ):
    # Normalized compression distance
    # https://en.wikipedia.org/wiki/Normalized_compression_distance

    return 0

def NGD(x, y):
    # Normalized Google distance
    # https://en.wikipedia.org/wiki/Normalized_Google_distance
    return 0



'''
Text distances
https://pypi.org/project/textdistance/

Parameters
----------
x: string
y: string

Result
------
result: float
    Distance between two strings
'''
import textdistance

# textdistance.hamming('test', 'text')
# textdistance.hamming.distance('test', 'text')
# textdistance.hamming.similarity('test', 'text')
# textdistance.hamming.normalized_distance('test', 'text')
# textdistance.hamming.normalized_similarity('test', 'text')
# textdistance.Hamming(qval=2).distance('test', 'text')




'''
Other
'''
def corr_cat(x, y, vc_x=None, vc_y=None):
    '''
    Categorical correlation
    using Weighted Jaccard Index
    https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    x: Series, 1st categorical column
    y: Series, 2nd categorical column

    Result
    ------
    result: float
        Similarity between two categorical columns
    '''
    xy_cols = ['x', 'y']
    renamer = dict(zip([x.name, y.name], xy_cols))
    xy = pd.concat([x, y], axis=1)
    xy = xy.rename(columns=renamer)

    vc_x = xy.x.value_counts() if vc_x is None else vc_x
    vc_y = xy.y.value_counts() if vc_y is None else vc_y

    xy_count = xy.groupby(xy_cols).size().reset_index(name='I')
    xy_count['weights'] = xy_count.I / sum(xy_count.I)
    xy_count['x_count'] = xy_count.x.map(vc_x)
    xy_count['y_count'] = xy_count.y.map(vc_y)
    xy_count['U'] = xy_count.x_count + xy_count.y_count - xy_count.I
    xy_count['IoU'] = xy_count.I / xy_count.U
    result = sum(xy_count['IoU'] * xy_count['weights'])
    return result
