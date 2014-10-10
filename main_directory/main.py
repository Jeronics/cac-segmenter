

__author__ = 'jeronicarandellsaladich'
import sys
import time
from utils import *

if __name__ == "__main__":
    start = time.time()

    # Read inputs, return numpy arrays for image, cage/s and mask.
    image, mask_file, init_cage_file, curr_cage_file=get_inputs(sys.argv)

    # If Image is an angle image, normalize..

    # Calculate Mean Value Coordinates
        # IN C PROGRAMMING
        # Get initial Contour From Initial Mask.
            # Add more points, smooth contour.
            # Calculate Angle of each point in the contour with the control points.

            # Calculate Distance to each control point
        # Calculate Coordinates(Formula)

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