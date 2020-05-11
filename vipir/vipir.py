from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timezone

import matplotlib
from matplotlib import cm
from PIL import Image, ImageDraw, ImageEnhance

from vipir.utils import modelutils as mod_util

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

    def image(self, which='total_power',size=(1024, 1024), enhance=True):

        snr = self.snr(which)
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
        return im.convert('RGB')

    def get_objects(self, model, thresh=0.7):
        mod = model['model']
        labs = model['labels']
        
        im = self.image()
        imsize = im.size
        image_np = np.array(im)
        # Actual detection.
        output_dict = mod_util.run_inference_for_single_image(mod, image_np)

        # loop through the returned objects and pick out those that exceed the threshold
        # relies on fact that output_dict['detection_score'] is sorted
        res = []
        for i in range(len(output_dict['detection_scores'])):
            if output_dict['detection_scores'][i] < thresh:
                break
            bbox_norm = output_dict['detection_boxes'][i]
            rngfreq = self.bbox_rngfreq(bbox_norm)
            obj = {'score': output_dict['detection_scores'][i],
                   'objtype': output_dict['detection_classes'][i],
                   'label':labs[output_dict['detection_classes'][i]]['name'],
                   'bbox_norm': bbox_norm,
                   'rngfreq': rngfreq
                    }

            res.append(obj)

        return res
    
    def bbox_rngfreq(self, bbox, normalized_coords=True):
        """
        takes a bounding box and returns corresponding coords in rng and freq
        and their indices in the rng and freq vectors
        """

        def pixtofreq(self, pix, sz):
            rate_pix = (self.maxfreq/self.minfreq)**(1/sz)
            val_pix = self.minfreq*(rate_pix**pix)
            rate_v = (self.maxfreq/self.minfreq)**(1/len(self.freq))
            indx = int((np.log10(val_pix)-np.log10(self.minfreq))/np.log10(rate_v))
            return self.freq[indx], indx

        def pixtorng(self, pix, sz):
            dpix = (self.maxrng-self.minrng)/float(sz)
            val_pix = self.minrng + dpix*pix
            dv = (self.maxrng-self.minrng)/len(self.rng)
            indx = int(np.floor((val_pix-self.minrng)/dv))
            return self.rng[indx], indx

        imsize = self.imsize
        # bbox originates from upper left
        if normalized_coords:
            ymin = (1-bbox[0])*imsize[0]; ymax = (1-bbox[2])*imsize[0]
            xmin = bbox[1]*imsize[1]; xmax = bbox[3]*imsize[1]
        else:
            ymin = imsize[0]-bbox[0]; ymax = imsize[0]-bbox[2]
            xmin = bbox[1]; xmax = bbox[3]

        minfreq, minfreq_i = pixtofreq(self, xmin, imsize[1])
        maxfreq, maxfreq_i = pixtofreq(self, xmax, imsize[1])
        #swap ymin and ymax because axis is flipped
        minrng, minrng_i = pixtorng(self, ymax, imsize[0])
        maxrng, maxrng_i = pixtorng(self, ymin, imsize[0])

        rngfreq = {'coords':{'minfreq':minfreq, 'maxfreq':maxfreq,
                             'minrng':minrng, 'maxrng':maxrng},
                   'indices':{'minfreq_i':minfreq_i, 'maxfreq_i':maxfreq_i,
                              'minrng_i': minrng_i, 'maxrng_i':maxrng_i}
                  }

        return rngfreq

    def get_objcontents(self, rngfreq_i):

        snr = self.snr()
        return (snr[rngfreq_i['minfreq_i']:rngfreq_i['maxfreq_i'],
                   rngfreq_i['minrng_i']:rngfreq_i['maxrng_i']],
                self.freq[rngfreq_i['minfreq_i']:rngfreq_i['maxfreq_i']],
                self.rng[rngfreq_i['minrng_i']:rngfreq_i['maxrng_i']])


