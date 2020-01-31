from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timezone

import matplotlib
from matplotlib import cm
from PIL import Image, ImageDraw, ImageEnhance

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

            self.total_power = np.asarray(rootgrp.variables['total_power'][:,:])
            self.total_noise = np.asarray(rootgrp.variables['total_noise'][:])
            
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

    # TODO: figure out a way to encapsulate the snr computation
    def snr(self):
        return self.total_power-self.total_noise.reshape(-1,1)

    def image(self, size=(1024, 1024), enhance=True):

        snr = self.total_power-self.total_noise.reshape(-1,1)
        cmap = matplotlib.cm.get_cmap('gnuplot')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        snr_np = cmap(norm(snr.T), bytes=True)
        snr_im = Image.fromarray(snr_np, mode='RGBA').transpose(Image.FLIP_TOP_BOTTOM)

        # Jack the contrast if asked
        if enhance:
            enh = ImageEnhance.Contrast(snr_im)
            im = enh.enhance(1.5)
        else:
            im = snr_im

        #make 'em all same size
        im = im.resize(size)
        self.imsize = im.size
        return im

    def pixel_to_coords(self, pxl):
        """
        returns the frequency, range values assoc'd with thepixel
        """
        ind_x = pxl[0]
        ind_y = self.nrng - pxl[1] #y axis flipped in image

        assert(ind_x >= 0 and ind_x < self.nfreq)
        assert(ind_y >= 0 and ind_y < self.nrng)

        return (self.freq[ind_x], self.rng[ind_y])

    def coords_to_pixel(self, coords, size=(1024, 1024)):
        """
        returns pixel coordinates of the frequency, range tuple
        """

        width, height = size

        f = coords[0]
        r = coords[1]

        assert(f >= self.minfreq and f <= self.maxfreq)
        assert(r >= self.minrng and r <= self.maxrng)

        #frequency is in log scale
        flogmin = np.log10(self.minfreq)
        flogmax = np.log10(self.maxfreq)
        flog = np.log10(f)
        f_pix = int(width*(flog - flogmin)/(flogmax-flogmin))

        # range dimension is inverted in the image
        r_pixup = int(height*(r-self.minrng)/(self.maxrng-self.minrng))
        # flip it
        r_pix = height - r_pixup
        
        return (f_pix, r_pix)

