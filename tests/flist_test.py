
import sys
import os
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

sys.path.append('./src')
from netCDF4 import Dataset

from vipir.vipir import vipir as vp, get_cdf, get_flist

def testlen():
    flist = get_flist('WI937','2020-07-31 18:00:00', '2020-07-31 18:30:00', interval=None)
    for i in range(len(flist)):
        print(f'{i}: {flist[i][0]}')

    assert len(flist)==15

#test for observations spaced at least n minutes apart, in this case 5
def testinteval():

    td = timedelta(minutes=5)
    #these are the time strings at 5 minute interval btwn 18:00 and 18:30 for this station
    ground_truth = ['2020-07-31 18:00:02','2020-07-31 18:06:03',
                    '2020-07-31 18:12:02','2020-07-31 18:18:03','2020-07-31 18:24:03']

    flist = get_flist('WI937','2020-07-31 18:00:00', '2020-07-31 18:30:00', interval=td)

    for f in flist:
        print(f'Returned: {f[0]}')

    assert len(flist) == len(ground_truth)

    #get the right ones?
    for f, g in zip(flist,ground_truth):
        assert f[0] == g


#test over a day boundary:
# make sure sampling stays spaced out properly
def testmultiday():
    td = timedelta(hours=1)
    ground_truth = ['2020-07-31 18:00:02','2020-07-31 19:00:02','2020-07-31 20:00:02',
                    '2020-07-31 21:00:03','2020-07-31 22:02:02','2020-07-31 23:02:02',
                    '2020-08-01 00:02:02','2020-08-01 01:02:02','2020-08-01 02:02:02',
                    '2020-08-01 03:02:02','2020-08-01 04:02:03','2020-08-01 05:04:03']

    flist = get_flist('WI937', '2020-07-31 18:00:00','2020-08-01 06:00:00', interval=td)

    for f in flist:
        print(f'Returned: {f[0]}')

    assert len(flist) == len(ground_truth)
    
    #get the right ones?
    for f, g in zip(flist,ground_truth):
        assert f[0] == g
