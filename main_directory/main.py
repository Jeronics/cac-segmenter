__author__ = 'jeronicarandellsaladich'
import sys
import time
from utils import *

if __name__ == "__main__":
    start=time.time()

    # Read inputs, return numpy arrays for cage/s and mask.
    mask_file, init_cage_file, curr_cage_file=get_inputs(sys.argv)
 
    # THE END

    # Time elapsed
    end=time.time()
    print end-start