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

    #testlibrary = CDLL("testlibrary.so")
    libcac=CDLL("apicac/libcac.so")

    cac_contour_get_interior_contour=libcac.cac_contour_get_interior_contour
    LP_c_int = POINTER(c_int) # mateixa notacio que python
    LP_c_double = POINTER(c_double) # mateixa notacio que python
    LP_LP_c_double = POINTER(LP_c_double) #  mateixa notacio que python

    # millor no posar-ho per "saltar-nos" un error que indica el python. No se com arreglar-lo
    # cac_contour_get_interior_contour.argtypes=[LP_c_int, LP_LP_c_double, LP_c_double, c_int, c_int, c_int]

    nrow, ncol = mask_file.shape
    img=np.copy(mask_file)
    #printNpArray(img)

    contour_size=c_int()  # un sencer
    mat = LP_c_double()   # un punter a double


    # Amb 'byref' passem la referencia a la variable
    # Amb as_types transformem de tipus numpy a tipus ctypes, cosa que va millor
    cac_contour_get_interior_contour(byref(contour_size), byref(mat), ctypeslib.as_ctypes(mask_file), c_int(ncol), c_int(nrow), c_int(4))

    # passem la matriu retornada a tipus numpy. Observa com defineixo la mida de la matriu
    contour_coordinates = ctypeslib.as_array(mat,shape=(contour_size.value,2));

    #OPTIONAL: PRINT THE CONTOUR ON THE IMAGE
    # plotContourOnImage(contour_coordinates,image)

    control_points=1
    affine_coordinates_of_contour = np.zeros([contour_coordinates.shape[0],control_points])

    # THE END
    # Time elapsed
    end = time.time()
    #print end-start


#TODO
# IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional
# C-TYPES Dynamic matrix