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

    testlibrary = CDLL("testlibrary.so")
    libcac=CDLL("apicac/libcac.so")

    cac_contour_get_interior_contour=testlibrary.cac_contour_get_interior_contour
    LP_c_int = POINTER(c_int) # mateixa notacio que python
    LP_c_double = POINTER(c_double) # mateixa notacio que python
    LP_LP_c_double = POINTER(LP_c_double) #  mateixa notacio que python

    # millor no posar-ho per "saltar-nos" un error que indica el python. No se com arreglar-lo
    # cac_contour_get_interior_contour.argtypes=[LP_c_int, LP_LP_c_double, LP_c_double, c_int, c_int, c_int]

    img = np.array([[ 0.1, 0.2, 0.2 ],[ 0.3, 0.4,9.3 ],[ 0.5, 0.6, 5.2 ]], dtype=np.float64, ndmin=2)

    print img.flags
    img.setflags(write=1,align=1)
    print img.flags
    #nrow, ncol = img.shape
    nrow, ncol = image[:,:,1].shape
    img=image[:,:,1]
    #printNpArray(img)



    contour_size=c_int()  # un sencer
    mat = LP_c_double()   # un punter a double

    # Amb 'byref' passem la referencia a la variable
    # Amb as_types transformem de tipus numpy a tipus ctypes, cosa que va millor
    cac_contour_get_interior_contour(byref(contour_size), byref(mat), ctypeslib.as_ctypes(img), c_int(ncol), c_int(nrow), c_int(4))

    # passem la matriu retornada a tipus numpy. Observa com defineixo la mida de la matriu
    matriu = ctypeslib.as_array(mat,shape=(contour_size.value,2));

    printNpArray(matriu)
    # THE END
    # Time elapsed
    end = time.time()
    #print end-start


#TODO
# IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional
# C-TYPES Dynamic matrix