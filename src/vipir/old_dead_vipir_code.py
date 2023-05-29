# where dead code goes to die
  def image(self, which='total_power',size=(1024, 1024), thresh=3, enhance=True):
        
        img_np = self.img_array(thresh=thresh)
        
        # build the image in the proper orientation
        snr_im = Image.fromarray(img_np, mode='RGBA').transpose(Image.FLIP_TOP_BOTTOM)

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
    
    def plot_obs(self, ax=None):
        
        cmaps = self.get_colormaps()
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        
        #get the plot axes
        freq = self.freq
        rng = self.rng
        
        #get the power measures
        o_pwr = self.snr(which='O_mode_power')
        x_pwr = self.snr(which='X_mode_power')
        
        # identify the cells in which x_pwr exceeds o_pwr by more than the threshold
        mask = x_pwr > o_pwr+thresh
        
        # mask out the o_ and x_pwr values
        x_pwr_m = masked_array(x_pwr, np.logical_not(mask))
        o_pwr_m = masked_array(o_pwr, mask)
        
        px = ax.pcolormesh(x_pwr_m.T,cmap=green_cmap,
               norm=colors.PowerNorm(gamma=gamma),
              vmin=0, vmax=50)
        
        po = ax.pcolormesh(o_pwr_m.T,cmap=red_cmap,
               norm=colors.PowerNorm(gamma=gamma),
              vmin=0, vmax=50)

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


   def get_trace_bbox(self, imthresh = 30.0, thresh=3.0, size=5):
        import scipy.ndimage as nd
        
        #make the image power array
        x_pwr = self.snr('X_mode_power')
        o_pwr = self.snr('O_mode_power')
        pwr = np.where(x_pwr >= o_pwr+thresh, x_pwr, o_pwr)

        #convolve (max) to increase the connectivity
        pwr = nd.generic_filter(pwr,np.max,size=size,mode='nearest')

        #mask out the backgound (and transpose)
        im = pwr.T >= imthresh

        #label and find the features in the masked image
        feats, nfeats = nd.label(im, structure=nd.generate_binary_structure(2,2))
        #get the object slices
        objs = nd.find_objects(feats)

        #calculate the areas of the objects and find the largest
        areas = np.array([ (s[0].stop-s[0].start)*(s[1].stop-s[1].start) for s in objs])
        b = areas.argmax() #index of the object covering largest area

        #dredge up the freq and range vals for the bbox
        bbox = {'freqstart': self.freq[objs[b][1].start],
                'freqend':   self.freq[objs[b][1].stop-1],
                'rngstart':  self.rng[ objs[b][0].start],
                'rngend':    self.rng[ objs[b][0].stop-1]}

        return bbox
