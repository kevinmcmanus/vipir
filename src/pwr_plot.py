import matplotlib.pyplot as plt

import sys
sys.path.append('./src')

from vipir.vipir import vipir as vp, get_cdf, get_flist, load_cdf

obslist = load_cdf(r'./data/WI937')
#    assert len(obslist) ==24
print(f'{len(obslist)} files loaded from cache')

fig = plt.figure(figsize=(24, 18))

obslist[-1].plot_pwr(fig)
plt.show()
