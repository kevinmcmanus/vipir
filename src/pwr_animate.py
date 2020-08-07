import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from datetime import datetime, timezone, timedelta

from numpy.ma import masked_array
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.colorbar import colorbar

import os
import sys
sys.path.append('./src')

from vipir.vipir import vipir as vp, get_cdf, get_flist, load_cdf

print(f'Current working directory: {os.getcwd()}')

#flist = get_flist('WI937', '2020-07-31 18:00:00','2020-08-01 06:00:00', interval=timedelta(hours=1))
obslist = load_cdf(r'../data/WI937cache')
print(f'{len(obslist)} files retrieved')

#useful data structure for animating
x_pwr_thresh = 3.0
pwr_gamma = 0.5

#normalization max & min
pwr_min = 0.0
pwr_max = 50.0

animation_data = {
    'colormaps': obslist[0].get_colormaps(),
    'pwr_min': pwr_min,
    'pwr_max': pwr_max,
    'pwr_norm':colors.PowerNorm(gamma=pwr_gamma), 
    'station': obslist[0].station,
    'freq': obslist[0].freq,
    'rng': obslist[0].rng,
    'obs_time':[o.obs_time for o in obslist],
    'o_pwr_masked':[masked_array(o.snr(which='O_mode_power'),
            o.snr(which='X_mode_power') > o.snr(which='O_mode_power')+x_pwr_thresh) for o in obslist],
    'x_pwr_masked':[masked_array(o.snr(which='X_mode_power'),
            o.snr(which='X_mode_power') <= o.snr(which='O_mode_power')+x_pwr_thresh) for o in obslist]
    }


fig = plt.figure(figsize=(12,9))
#set up the figure
gs = gridspec.GridSpec(1,3, width_ratios=[8,1,1])
ax = fig.add_subplot(gs[0], frameon=True)

ax.set_xscale('log')
ax.set_ylabel('Range (km)')
ax.set_xlabel('Frequency log(kHz)')
ax.set_title(animation_data['station'] + ' ' +animation_data['obs_time'][0].strftime('%Y-%m-%d %H:%M:%S %Z'))

#mesh the x_power
px = ax.pcolormesh(animation_data['freq'],
                   animation_data['rng'],
                   animation_data['x_pwr_masked'][0].T,
                   cmap=animation_data['colormaps']['x_pwr_cmap'],
                   norm=animation_data['pwr_norm'],
                   vmin = animation_data['pwr_min'],
                   vmax = animation_data['pwr_max']
                    )
#x power colorbar
cax_x = fig.add_subplot(gs[1],frameon=False)
cba = plt.colorbar(px, cax=cax_x, shrink=0.75, fraction=0.5,
                    extend='max')
cax_x.set_title('X-Power', pad=0.1)

#mesh the o power:
po = ax.pcolormesh(animation_data['freq'],
                   animation_data['rng'],
                   animation_data['o_pwr_masked'][0].T,
                   cmap=animation_data['colormaps']['o_pwr_cmap'],
                   norm=animation_data['pwr_norm'],
                   vmin = animation_data['pwr_min'],
                   vmax = animation_data['pwr_max']
                    )
cax_o = fig.add_subplot(gs[2],frameon=False)
cbb = plt.colorbar(po, cax=cax_o,shrink=0.25, fraction=0.5,
                    extend='max')
cax_o.set_title('O-Power')
cax_o.set_ylabel('SNR (dB)', rotation=270, labelpad=0.15)

def init():
    quad1.set_array([])
    return quad1,

def animate(iter):
    stn = animation_data['station']
    obs_time = animation_data['obs_time'][iter]
    ax.set_title(stn + ' ' + obs_time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    px.set_array(animation_data['x_pwr_masked'][iter][:-1,:-1].T.flatten())
    po.set_array(animation_data['o_pwr_masked'][iter][:-1,:-1].T.flatten())
    return px, po

#plt.ion()

anim = animation.FuncAnimation(fig,animate,frames=len(obslist),interval=200,blit=False,
        repeat=True, repeat_delay=2000)
plt.show()

print ('Finished!!')