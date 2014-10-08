__author__ = 'jeronicarandellsaladich'
import sys
import time
from utils import *

if __name__ == "__main__":
    start = time.time()

    # Read inputs, return numpy arrays for image, cage/s and mask.
    image, mask_file, init_cage_file, curr_cage_file=get_inputs(sys.argv)
    # WHILE

        # Define inner and outer points Omega1 and Omega2
            #   Omega1: From Mask ---> calculate contour
            #   Omega2: - Whole\Omega1 --- (subcase of the next:)
            #           - Dilation(Omega1)\Omega1 ---> Distance

        # Define Energy function
            #   Create Gradient Of energy function
            #       Mean Model
            #       Gaussian Model
            #       Histogram Model

    # THE END
    # Time elapsed
    end = time.time()
    print end-start