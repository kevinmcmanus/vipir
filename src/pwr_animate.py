import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.animation as animation
import argparse

import sys
sys.path.append('./src')

#https://stackoverflow.com/questions/21487916/animating-a-quadmesh-from-pcolormesh-with-matplotlib


from vipir.vipir import vipir as vp, get_cdf, get_flist, load_cdf, get_cdflist

def update(obsno):
    obs=vp(obslist[obsno])
    img = obs.img_array()
    imp.set_data(img)
    ax.set_title(obs.station + ' ' +obs.obs_time.strftime('%Y-%m-%d %H:%M:%S %Z'))

parser = argparse.ArgumentParser(description='Runs animation of netCDF files in directory')
parser.add_argument('--dir',help='sample interval')


args = parser.parse_args()
dir = args.dir if args.dir is not None else r'./data/WI937'

obslist = get_cdflist(dir)
#    assert len(obslist) ==24
print(f'{len(obslist)} files loaded from cache')


fig, ax = plt.subplots(figsize=(24, 18))
#cmaps = obslist[0].get_colormaps()



obs = vp(obslist[0])
img = obs.img_array()
ext = [0, 1,obs.rng.min(), obs.rng.max()]
imp = ax.imshow(img, origin='lower', aspect='auto', extent = ext)
l = len(obs.freq)-1
ticks = ax.get_xticks()
ax.set_xticks(ax.get_xticks().tolist())
ax.set_xticklabels([f'{obs.freq[int(l*f)]:.1f}' for f in ticks])
ax.grid(color='grey', ls=':', lw=1)

anim = animation.FuncAnimation(fig, update, frames=range(len(obslist)),
                               interval=500, repeat_delay=2000)
plt.show()


