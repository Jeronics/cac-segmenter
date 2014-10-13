

__author__ = 'jeronicarandellsaladich'

import time
from utils import *
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
if __name__ == "__main__":
    start = time.time()

    # Read inputs, return numpy arrays for image, cage/s and mask.
    image, mask_file, init_cage_file, curr_cage_file = get_inputs(sys.argv)

    # If Image is an angle image, normalize..

    # Calculate Mean Value Coordinates
        # IN C PROGRAMMING
        # Get initial Contour From Initial Mask.
            # Add more points, smooth contour.
            # Calculate Angle of each point in the contour with the control points.

            # Calculate Distance to each control point
        # Calculate Coordinates(Formula)
    '''
        void cac_contour_get_interior_contour(
            int *contour_size,      /* output */
            float **contour_coord,  /* output */
            float **img,             /* input */
            int ncol,               /* input */
            int nrow,               /* input */
            int conn)               /* input */
    '''
    testlibrary = CDLL("testlibrary.so")
    cac_contour_get_interior_contour=testlibrary.cac_contour_get_interior_contour

    c_float_p = POINTER(c_float)
    c_float_p_p = POINTER(c_float_p)
    img = np.array([[ 0.1, 0.2 ],[ 0.3, 0.4 ]], dtype=np.float, ndmin=2)
    contour_coord=np.empty([1,1],dtype=float)
    contour_size=c_int(0)
    nrow, ncol = img.shape
    c_float_p=POINTER(c_float)
    c_float_p_p=POINTER(c_float_p)
    nrow,ncol=img.shape
    conn=c_int(3)

    cac_contour_get_interior_contour.argtypes=[POINTER(c_int), POINTER(c_float_p_p), POINTER(c_float_p_p),c_int, c_int, c_int]
    cac_contour_get_interior_contour(contour_size, contour_coord.ctypes.data_as(c_float_p_p),img.ctypes.data_as(c_float_p_p), ncol,nrow, conn)
    print 'Hi'
    print img;

    # WHILE    Vertex_previous-Vertex_new > tol     &&      iterations<MaxNumIterations

        # IN C PROGRAMMING
        # ENTRADA: Contorn en python
        # SORTIDA:
            # - Llista pixels interns
            # - Llista pixels externs
            # - Coordenades afins pixels Interns
            # - Coordenades afins pixels Externs
        # Re-Define inner and outer points Omega1 and Omega2
            #   Recalculate Contour:
            #   Omega1: Distance function from the contour
            #   Omega2: Dilation(Omega1)\Omega1 ---> Distance

        # Re-define Energy function
            #   Create Gradient Of energy function

            #       Mean Model <----------------- IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional
            #       Gaussian Model
            #       Histogram Model

        # Minimize Energy function:
            #   Gradient decent on the gradient of the energy function
                #   Restriction: Vertex V_i runs along (P_m-V_i) line. (P_m is the mass point)
                #   IF the minimization step gets stuck (sign alternates):
                    #   Change Method --> Minimize Step
                #   ELSE
                    #   Keep a constant step in the minimization algorithm

    # THE END
    # Time elapsed
    end = time.time()
    print end-start

#TODO
# IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional
# C-TYPES Dynamic matrix