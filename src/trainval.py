#
"""
 creates list.txt, train.txt and val.txt as req'd by create tfrecord
 also fixes up xmls to put correct path in and creates mask file
"""

import os
from os.path import join
import shutil
import numpy as np
import re
from xml.etree import ElementTree as ET
from PIL import Image

# input images and xmls in ./model_images and ./model_xmls resp

PROJECT_ROOT = os.getcwd()
# where the inputs are coming from 
ALLIMAGEDIR = join(PROJECT_ROOT,'model_images')
ALLXMLDIR = join(PROJECT_ROOT, 'model_xmls')

#where the outputs are going
ANNOTATIONSDIR = join(PROJECT_ROOT,'annotations')
XMLDIR = join(ANNOTATIONSDIR, 'xmls')
IMAGEDEST = 'images' # destination folder for images
IMAGEDIR = join(ANNOTATIONSDIR, IMAGEDEST)  # where the images wind up
TRIMAPDIR = join(ANNOTATIONSDIR, 'trimaps')

def xmltodict(xmlfile):
    retdict = {}
    xml = ET.parse(join(ALLXMLDIR, xmlfile))
    root_element = xml.getroot()
    retdict['folder'] = root_element.find('folder').text
    retdict['basename'] = root_element.find('filename').text.split('.')[0] # just the basename
    objs = root_element.findall('object')
    # better just be one
    assert(len(objs)==1)
    retdict['name'] = objs[0].find('name').text

    #get the bounding box
    retdict['bndbox'] = {}
    bbox = objs[0].find('bndbox')
    retdict['bndbox']['xmin'] = int(bbox.find('xmin').text)
    retdict['bndbox']['ymin'] = int(bbox.find('ymin').text)
    retdict['bndbox']['xmax'] = int(bbox.find('xmax').text)
    retdict['bndbox']['ymax'] = int(bbox.find('ymax').text)

    return retdict

#loop though the xml files and fix the xml, create the mask and copy the image
def copyimagefile(xmldict):
    
    fname = xmldict['basename']+'.jpg'
    shutil.copyfile(ALLIMAGEDIR+'/'+fname,
        IMAGEDIR+'/'+fname)


def createpartitions(xmls):

    #partition into training and validation lists
    # make a 70/30 split
    np.random.seed(1234)
    thresh = 0.7
    vals = np.random.uniform(size=len(xmls))

    #create the list files and write them out
    list_path = join(ANNOTATIONSDIR,'list.txt')
    #files to hold partition into trainval and test sets
    trainval_path = join(ANNOTATIONSDIR,'trainval.txt')
    test_path = join(ANNOTATIONSDIR,'test.txt')

    with open(list_path,'w') as listf, open(trainval_path,'w') as tvf,open(test_path, 'w') as testf:
        for f,v  in zip(xmls,vals):
            #TODO get these out label text file
            obj_class = 1 if f['name'] == 'trace' else 2

            listf.write('{0} {1} {2} {3}\n'.format(f['basename'],obj_class,1,1))

            fout = tvf if v < thresh else testf
            fout.write('{0} {1} {2} {3}\n'.format(f['basename'],obj_class,1,1))

def fixandwritexml(xmldict):

    #get the orig xml
    xml = ET.parse(join(ALLXMLDIR, xmldict['basename']+'.xml'))
    root_element = xml.getroot()

    #update the path element
    path_element = root_element.find('path')
    path_element.text = join(IMAGEDIR, xmldict['basename']+'.jpg')

    #fix up the folder element
    folder_element = root_element.find('folder')
    folder_element.text = IMAGEDEST

    #write it out to xml directory
    xml.write(join(XMLDIR, xmldict['basename']+'.xml'))


def createtrimap(xmldict):

    mask = make_mask(xmldict)

    trimappath = join(TRIMAPDIR,xmldict['basename']+'.png')

    mask.save(trimappath, mode='PNG')



def make_mask(xmldict):
    
    img=Image.open(join(ALLIMAGEDIR, xmldict['basename']+'.jpg'))
    imgsz = img.size
    
    bbox = xmldict['bndbox']

    minx = bbox['xmin']
    miny = bbox['ymin']
    maxx = bbox['xmax']
    maxy = bbox['ymax']
    
    # note y values correspond to the rows of the image, x values to the columns
    
    mask_np = np.zeros((imgsz[0], imgsz[1]), dtype=np.uint8)
    mask_np[ miny:maxy, minx:maxx] = 1.0

    mask = Image.fromarray(mask_np)
    
    return mask


if __name__ == "__main__":

    #blow away existing annotations directory
    shutil.rmtree(ANNOTATIONSDIR,ignore_errors=True)

    # make new directories for everything
    for d in [ANNOTATIONSDIR, IMAGEDIR, XMLDIR, TRIMAPDIR]:
        try:
            os.mkdir(d)
        except OSError:
            print ("Creation of the directory %s failed" % d)
        else:
            print ("Successfully created the directory %s " % d)

    #get list of examples from input xmls directory
    xmldicts = [xmltodict(f) for f in os.listdir(ALLXMLDIR) if re.match('.*\.xml',f)]

    #process each xml
    for xml in xmldicts:
        copyimagefile(xml)
        createtrimap(xml)
        fixandwritexml(xml)

    #create trainval and test partitions
    createpartitions(xmldicts)
