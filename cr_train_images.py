import os
from os.path import isfile, join
import re
import vipir as vp
from shutil import rmtree
from PIL import Image


cdfdir = './netcdf'
imdir = './model_images'

cdffiles = [f for f in os.listdir(cdfdir)
        if isfile(join(cdfdir, f)) and re.search('.*\.NGI$',f)]

print(f'Files found: {len(cdffiles)}')

#set up the output directory
rmtree(imdir, ignore_errors=True)
os.mkdir(imdir)

for cdf in cdffiles:
    vip = vp.vipir(join(cdfdir,cdf))
    im = vip.image()
    imout = re.sub('\.NGI$','.jpg', cdf)
    im.convert('RGB').save(join(imdir, imout),'JPEG')
