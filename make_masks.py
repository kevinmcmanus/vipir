#script to produce image masks
import os
import shutil
import numpy as np
import imageio
from xml.etree import ElementTree as ET

XMLDIR = './annotations/xmls_new'
TRIMAPDIR = './annotations/trimaps'
import shutil
def get_bounding_boxes(xml):
    root_el = xml.getroot()
    objs = root_el.findall('object')
    boxes = {}
    for i,o in enumerate(objs):
        bbox = o.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes['box'+str(i)] = {'Origin':(xmin, ymin),
                              'xextent':xmax-xmin, 'yextent':ymax-ymin}
    return boxes

def getimgpath(imgxml):
    root_element = imgxml.getroot()
    folder = root_element.find('folder')
    filename = root_element.find('filename')
    
    return ('{0}/{1}'.format(folder.text, filename.text),
        filename.text.split('.')[0]) #the basename

def make_mask(imgxml):
    
    imgpath, bname = getimgpath(imgxml)
    img=imageio.imread(imgpath)
    imgsz = img.shape
    
    bbox = get_bounding_boxes(imgxml) 
    assert(len(bbox)==1)
    bb = bbox['box0']
    
    minx = bb['Origin'][0]
    miny = bb['Origin'][1]
    maxx = bb['xextent'] + minx
    maxy = bb['yextent'] + miny
    
    # note y values correspond to the rows of the image, x values to the columns
    
    mask = np.zeros((imgsz[0], imgsz[1]), dtype=np.uint8)
    mask[ miny:maxy, minx:maxx] = 1.0
    
    return mask, bname

#create the output directory
shutil.rmtree(TRIMAPDIR,ignore_errors=True)
path = TRIMAPDIR
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

#get list of xmls (need to make a mask for each xml)
xmls = [f for f in os.listdir(XMLDIR)]
imagexmls = [ET.parse(XMLDIR + os.sep + xmlfile) \
             for xmlfile in xmls]

for imgxml in imagexmls:
    mask, bname = make_mask(imgxml)
    imageio.imwrite(TRIMAPDIR + os.sep + bname + '.png',mask)
