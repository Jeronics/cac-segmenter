__author__ = 'jeronicarandellsaladich'

import re
import numpy as np
from matplotlib import pyplot


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def get_inputs(arguments):
    """Return imagage, cage/s and mask from input as numpy array.
        Specification:  ./cac model imatge mascara caixa_init [caixa_curr]
    """

    if (len(arguments) != 6 and len(arguments) != 5 ):
        print 'Wrong Use!!!! Expected Input ' +sys.argv[0] + ' model(int) image(int) mask(int) init_cage(int) [curr_cage(int)]'
        sys.exit(1)

    for arg in arguments:
        print arg
    model = arguments[0]
    mask = int(arguments[1])
    init_cage= int(arguments[2])
    if len(arguments) == 6:
        curr_cage = int(arguments[5]);
    else:
        curr_cage = None


    # PATHS
    test_path = r'../test/elefant/'
    mask_num = '%(number)02d' % {"number": mask}
    init_cage_name = '%(number)02d' % {"number": init_cage}
    curr_cage_name = '%(number)02d' % {"number": curr_cage}

    image_name = test_path + 'image'+'.pgm'
    mask_name = test_path + 'mask_'+mask_num+'.pgm'
    init_cage_name = test_path + 'cage_'+init_cage_name+'.txt'
    curr_cage_name = test_path + 'cage_'+curr_cage_name+'.txt'

    # LOAD Cage/s and Mask
    image = read_pgm(image_name,byteorder='>')
    mask_file = read_pgm(mask_name,byteorder='>')
    init_cage_file = np.loadtxt(init_cage_name, float)
    curr_cage_file = np.loadtxt(curr_cage_name, float)

    return image, mask_file, init_cage_file, curr_cage_file



#TODO
    #Printing pgm files
        #pyplot.imshow(mask_file, pyplot.cm.gray)
        #pyplot.show()