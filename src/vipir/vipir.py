from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timezone
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from PIL import Image, ImageDraw, ImageEnhance

from numpy.ma import masked_array
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1.colorbar import colorbar

#from vipir.utils import modelutils as mod_util

class vipir():

    def __init__(self, path):
        
        with Dataset(path, "r", format="NETCDF4") as rootgrp:

            vars = rootgrp.variables

            self.station=np.asarray(vars['StationName'][:]).tostring().decode('UTF-8').strip()

            self.rng = np.asarray(rootgrp.variables['Range'][:])
            self.freq = np.asarray(rootgrp.variables['Frequency'][:])

            self.minrng = self.rng.min()
            self.maxrng = self.rng.max()
            self.nrng = len(self.rng)

            self.minfreq = self.freq.min()
            self.maxfreq = self.freq.max()
            self.nfreq = len(self.freq)
            
            #default image size (gets overwritten if image method called)
            self.imsize = (1024, 1024)

            # power and noise
            self.total_power = np.asarray(rootgrp.variables['total_power'][:,:])
            self.total_noise = np.asarray(rootgrp.variables['total_noise'][:])
            
            self.O_mode_power = np.asarray(rootgrp.variables['O-mode_power'][:,:])
            self.O_mode_noise = np.asarray(rootgrp.variables['O-mode_noise'][:])
            
            self.X_mode_power = np.asarray(rootgrp.variables['X-mode_power'][:,:])
            self.X_mode_noise = np.asarray(rootgrp.variables['X-mode_noise'][:])
            
            
            # some metadata
            self.obs_time = datetime(
                np.array(vars['year'][:]).item(),
                np.array(vars['month'][:]).item(),
                np.array(vars['day'][:]).item(),
                np.array(vars['hour'][:]).item(),
                np.array(vars['minute'][:]).item(),
                np.array(vars['second'][:]).item(),
                tzinfo=timezone.utc
            )

            self.station_location = {
                'longitude':np.array(vars['longitude'][:]).item(),
                'latitude':np.array(vars['latitude'][:]).item(),
                'altitude':np.array(vars['altitude'][:]).item()
            }

            self.colormaps = None


    # TODO: figure out a way to encapsulate the snr computation
    def snr(self, which='total_power', thresh=3):
        if which == 'total_power':
            pwr = self.total_power
            noise = self.total_noise
        elif which == 'O_mode_power':
            pwr = self.O_mode_power
            noise = self.O_mode_noise
        elif which == 'X_mode_power':
            pwr = self.X_mode_power
            noise = self.X_mode_noise
        elif which == 'O_mode_power_adj':
            raise ValueError('O_mode_power_adj no longer supported')
            X_mode_snr = self.X_mode_power - self.X_mode_noise.reshape(-1,1)
            O_mode_snr = self.O_mode_power - self.O_mode_noise.reshape(-1,1)
            pwr = np.where(X_mode_snr > O_mode_snr + thresh, X_mode_snr, O_mode_snr)
            
            #fake the noise vector
            noise = np.zeros(self.O_mode_noise.shape, dtype=float)
        else:
            s = ('Valid \'which\' parameter values are \'total_power\''
                    f', \'O_mode_power\', \'O_mode_power_adj\', and \'X_mode_power\'; which: {which}')
            raise ValueError(s)
            
        return pwr-noise.reshape(-1,1)
    
    def img_array(self, thresh=3, gamma=0.5):
        cmaps = self.get_colormaps()
        norm = colors.PowerNorm(vmin=0, vmax=50, gamma=gamma)
        
        #get the power measures
        o_pwr = self.snr(which='O_mode_power')
        x_pwr = self.snr(which='X_mode_power')
        
        # identify the cells in which x_pwr exceeds o_pwr by more than the threshold
        mask = x_pwr > o_pwr+thresh
        
        # mask out the o_ and x_pwr values
        x_pwr = x_pwr*mask
        o_pwr = o_pwr*np.logical_not(mask)

        o_np = cmaps['o_pwr_cmap'](norm(o_pwr),bytes=True)
        x_np = cmaps['x_pwr_cmap'](norm(x_pwr),bytes=True)
        
        #put them together and transpose
        img_np = np.transpose(o_np + x_np, axes=[1,0,2])
        
        return img_np

  
    def get_colormaps(self):
        if self.colormaps is None:
            self.set_colormaps(self.gen_colormaps())
        return self.colormaps
    
    def set_colormaps(self, cmaps):
        self.colormaps = cmaps
        
    def gen_colormaps(self):
        """
        generates rbga colormaps equivalents for red and green HSV cmaps
        """
        from matplotlib.colors import ListedColormap
        
        #this should probably be somewhere else!
        #references: Equations: (https://www.rapidtables.com/convert/color/hsv-to-rgb.html)
        #            Good Article: (http://colorizer.org/)

        def hsv_to_rgb(h,s,v):
            """
            converts hue-saturation-value (hsv) to rgb (rgba, actually)
            """
            c = v*s
            x = c*(1-np.abs(((h/60.0) % 2)-1))
            m = v-c

            rgb_tbl = np.array([[c,x,0],
                              [x,c,0],
                              [0,c,x],
                              [0,x,c],
                              [x,0,c],
                              [c,0,x]])

            rgb_prime =  rgb_tbl[int(h//60.0)]

            rgb = (rgb_prime+m)  #*255

            rgba = np.array([*rgb,1])

            return rgba
        
        N = 256
        sv = np.linspace(0.0, 1.0, N)
        
        # red is 0.0 degree hue, green is 90.0 degree hue in hsv scheme
        # use 1.0 for sat value, color map walks up the outer edge of the cone
        red_hsv   = np.array([hsv_to_rgb( 0.0, 1.0, v)for v in sv])
        green_hsv = np.array([hsv_to_rgb(90.0, 1.0, v)for v in sv])
        
        #make the color maps
        red_hsv_cmap   = ListedColormap(red_hsv)
        green_hsv_cmap = ListedColormap(green_hsv)
        
        return {'o_pwr_cmap':red_hsv_cmap, "x_pwr_cmap":green_hsv_cmap}


    def plot_pwr(self, fig, thresh=3.0, gamma=0.5):

        #x_power and o_power plotted using different color scales
        #plot o_power when it exceeds x_power by a threshold db
        o_pwr = self.snr('O_mode_power')
        x_pwr = self.snr('X_mode_power')
        #mask out the values that are NOT to plotted for each power
        x_pwr_m = masked_array(x_pwr, x_pwr <=  o_pwr+thresh)
        o_pwr_m = masked_array(o_pwr, x_pwr >   o_pwr+thresh)

        freq = self.freq
        rng = self.rng

        #set up the figure
        gs = gridspec.GridSpec(1,3, width_ratios=[18,1,1])
        ax = fig.add_subplot(gs[0], frameon=True)

        #colormaps
        cmaps = self.get_colormaps()

        #set up normalization
        norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=50)

        #plot the x_power and its colorbar
        px = ax.pcolormesh(freq, rng, x_pwr_m.T,cmap=cmaps['x_pwr_cmap'],
                   norm=norm)
        cax_x = fig.add_subplot(gs[1],frameon=False)
        cba = plt.colorbar(px, ax=cax_x, shrink=0.75, fraction=0.5,
                           extend='max')
        cax_x.xaxis.set_ticks([]); cax_x.yaxis.set_ticks([])
        cba.ax.set_xlabel('SNR\n(db)')
        cba.ax.set_title('X-Power')

        #plot the o_power and its colorbar
        po = ax.pcolormesh(freq, rng, o_pwr_m.T,cmap=cmaps['o_pwr_cmap'],
                       norm=norm)
        cax_o = fig.add_subplot(gs[2],frameon=False)
        cbb = plt.colorbar(po, ax=cax_o,shrink=0.75, fraction=0.5,
                         extend='max')
        cax_o.xaxis.set_ticks([]); cax_o.yaxis.set_ticks([])
        cbb.ax.set_xlabel('SNR\n(db)')
        cbb.ax.set_title('O-Power')

        #fix up the axes and annotation
        ax.set_xscale('log')
        ax.set_xticks(freq[0::len(freq)//8])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel('Frequency (kHz)')

        ax.set_ylabel('Range (km)')

        ax.set_title(self.station + ' ' +self.obs_time.strftime('%Y-%m-%d %H:%M:%S %Z'))
        ax.grid(color='grey', ls=':', lw=1)
        
        return ax

 
import tempfile
from ftplib import FTP
import os

def get_cdf(stn=None,dtstr=None, path=None, ftpsite ='ftp.ngdc.noaa.gov', rootdir='/ionosonde/data'):
    """
    Returns a vip for a station and date-time string
    """
    
    if path is None:
        yr = dtstr[0:4]
        daynum=dtstr[4:7]

        cdfpath = f'{rootdir}/{stn}/individual/{yr}/{daynum}/ionogram/{stn}_{dtstr}.NGI'
    else:
        cdfpath = path
        
    # create a temp file to fetch the cdf into
    fout = None
    fh =  tempfile.mkstemp()

    #fetch the cdf
    with FTP(ftpsite) as ftp, open(fh[1],'wb') as fout:
        ftp.login()
        ftp.retrbinary(f'RETR {cdfpath}', fout.write)
            
    # read up as a vip what we just ftp'd        
    vip = vipir(fh[1])
    
    #ditch the temp file
    os.close(fh[0]) # need to close since mkstemp opened it
    os.remove(fh[1])
    
    return vip

from datetime import datetime, timezone, timedelta
import re
def obsname_todt(obsname:str) -> datetime:
    year = int(obsname[6:10])
    daynum = int(obsname[10:13]) # day of the year
    hr = int(obsname[13:15])
    min = int(obsname[15:17])
    sec = int(obsname[17:19])
    
    dt = datetime(year, 1,1, hr, min, sec,tzinfo=timezone.utc )+timedelta(daynum-1)
    
    return dt



def get_flist(stn, fromstr, tostr, interval = timedelta(minutes=10),
              ftpsite ='ftp.ngdc.noaa.gov',rootdir='/ionosonde/data'):
    """
    returns list of ionogram (ngi) files from station over the 'fromstr' 'tostr' period separated
    by at least 'interval'.
    stn is station name, string
    fromstr and tostr are strings in iso format in utc
    interval is a timedelta, default 10 minutes, None=> 0 minutes, i.e. all observations
    """
    
    #convert fromstr and tostr to UTC  datetime objects with tzinfo
    fromdt = datetime.strptime(fromstr,'%Y-%m-%d %H:%M:%S')
    fromdt = fromdt.replace(tzinfo=timezone.utc)  
    todt = datetime.strptime(tostr,'%Y-%m-%d %H:%M:%S')
    todt = todt.replace(tzinfo=timezone.utc)

    #initialize lastdt to really early so that we pick up first observation in interval    
    lastdt = datetime.strptime('1900-01-01 00:00:00','%Y-%m-%d %H:%M:%S')
    lastdt = lastdt.replace(tzinfo=timezone.utc)

    if interval is None:
        interval = timedelta(minutes=0)

    #set up ftp connection
    ftp = FTP(ftpsite)
    ftp.login()
    
    #loop over the days and accumulate the qualifying file names

    thisdt = fromdt
    flist = []
    try:
        while thisdt <= todt:

            #change to the data directory for the current day
            yr = thisdt.year
            daynum =  thisdt.timetuple().tm_yday
            dir = f'{rootdir}/{stn}/individual/{yr}/{daynum:03}/ionogram/'
            ftp.cwd(dir)

            #get the NGI files and extract their date time from their names
            obs_list = [f for f in ftp.nlst() if re.search(r'^.*.NGI$',f)]
            obs_list.sort() # make sure in lexo/chrono order
            # get the datetimes from the file names
            obs_dt = [obsname_todt(o) for o in obs_list]

            #filter the file name list to be between fromdt and todt
            obs_all = [(f,o) for f,o in zip(obs_list,obs_dt) if o >= fromdt and o <= todt]
            
            #space out by the interval
            #find the next file that is at least 'interval' later than its predecessor
            obs_list = []
            for oa in obs_all:
                if (oa[1]-lastdt) >= interval:
                    obs_list.append((oa[1].strftime('%Y-%m-%d %H:%M:%S'), dir + oa[0]))
                    lastdt = oa[1]

            #accumulate the filtered file names
            flist += obs_list

            #see what tomorrow brings
            thisdt = thisdt.replace(hour=0, minute=0, second=0) + timedelta(days=1)
    
    finally:
        ftp.close()
    
    return flist

def load_cdf(dirname):
    """
    loads all of the NGI files in a directory, returns them in a list
    """

    flist = [f for f in os.listdir(dirname) if re.search(r'^.*.NGI$',f)]
    flist.sort()

    obslist = [vipir(os.path.join(dirname, f)) for f in flist ]

    return obslist

def get_cdflist(dirname):
    """
    finds all of the NGI files in a directory, returns their names in a list
    """

    flist = [f for f in os.listdir(dirname) if re.search(r'^.*.NGI$',f)]
    flist.sort()

    obslist = [os.path.join(dirname, f) for f in flist ]

    return obslist
        
if __name__ == '__main__':

    td = timedelta(hours=1)

    # #flist = get_flist('WI937', '2021-12-13 00:00:00','2021-12-14 23:59:59', interval=td)

    # for f in flist:
    #     print(f'Returned: {f[0]}')

    # assert len(flist) == 12

    obslist = load_cdf(r'.\data\WI937')
    #ssert len(obslist) ==24
    print(f'{len(obslist)} files loaded from cache')
    fig = plt.figure(figsize=(24, 18))
    obslist[-1].plot_pwr(fig)
    plt.show()