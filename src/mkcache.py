import os
import argparse
from datetime import datetime, timedelta
import sys
sys.path.append('./src')

from vipir.vipir import vipir as vp, get_cdf, get_flist
from ftplib import FTP

def ftp_cdf(cdfpath, dest,ftpsite ='ftp.ngdc.noaa.gov'):
        
    #fetch the cdf
    with FTP(ftpsite) as ftp, open(dest,'wb') as fout:
        ftp.login()
        ftp.retrbinary(f'RETR {cdfpath}', fout.write)

# command line args
parser = argparse.ArgumentParser(description='Build a cache of netCDF files')
parser.add_argument('stn', help='Station Name')
parser.add_argument('fromdt',help='start timestamp (UT)')
parser.add_argument('todt', help='end timestamp (UT)')
parser.add_argument('dirname', help='destination directory')
parser.add_argument('--interval',help='sample interval')

if __name__ == '__main__':


    args = parser.parse_args()

    #get command line args
    stn = args.stn

    #note use of 'T' in below format specifier
    from_dt = datetime.strptime(args.fromdt,'%Y-%m-%dT%H:%M:%S')
    to_dt = datetime.strptime(args.todt,'%Y-%m-%dT%H:%M:%S')

    interval = args.interval if args.interval is not None else 15 # 15 minute interval
    intervaltd = timedelta(minutes=int(interval))

    destdir = args.dirname

    #get list of qualifying files:
    flist = get_flist(stn, from_dt.strftime('%Y-%m-%d %H:%M:%S'),to_dt.strftime('%Y-%m-%d %H:%M:%S'), interval=intervaltd)
    print(f'Get_flist returned {len(flist)} files')

    os.mkdir(destdir)
    for f in flist:
        ftp_cdf(f[1], os.path.join(destdir, os.path.basename(f[1])))
        print(f'Copied: {os.path.basename(f[1])}')
   