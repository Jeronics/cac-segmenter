__author__ = 'jeronicarandellsaladich'
import sys
import time
from utils import *

if __name__ == "__main__":
    start=time.time()

    # Read inputs, return numpy arrays for image, cage/s and mask.
    image, mask_file, init_cage_file, curr_cage_file=get_inputs(sys.argv)

    # Define inner and outer points Omega1 and Omega2

    # 

    # THE END


    # Time elapsed
    end=time.time()
    print end-start