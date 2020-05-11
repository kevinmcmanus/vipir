import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET
from os.path import join

def annotationtodict(xmlfile, model_name, DATADIR='data'):
    retdict = {}
    ANNOTATIONSDIR = join(DATADIR, 'annotations')
    xml = ET.parse(join(model_name, ANNOTATIONSDIR, 'xmls', xmlfile + '.xml'))
    root_element = xml.getroot()
    retdict['folder'] = root_element.find('folder').text
    retdict['basename'] = root_element.find('filename').text.split('.')[0] # just the basename
    objs = root_element.findall('object')
    # better just be one
    assert(len(objs)==1)
    retdict['name'] = objs[0].find('name').text
    
    #get the corresponding image size
    IMAGEDIR = join(DATADIR,'images')
    im = Image.open(join(model_name, IMAGEDIR, xmlfile + '.jpg'))
    height = float(im.height)
    width = float(im.width)

    #get the bounding box and normalize it
    bbox = objs[0].find('bndbox')
    retdict['bndbox'] = np.array([
        float(bbox.find('ymin').text)/height,
        float(bbox.find('xmin').text)/width,
        float(bbox.find('ymax').text)/height,
        float(bbox.find('xmax').text)/width
        ])

    return retdict