#include <stdio.h>
#include <stdlib.h>
/*

gcc c_code_tests.c -shared -o testlibrary.so -fPIC

    void cac_contour_get_interior_contour(
        int *contour_size,      // output
        float **contour_coord,  // output
        float *img,             // input
        int ncol,               // input
        int nrow,               // input
        int conn)               // input

*/
void cac_contour_get_interior_contour(  int* pSize, double ** ppMem, double **image, int ncol, int nrow, int conn)
{
  int i, j;

  * pSize = 4;
  * ppMem = malloc( * pSize );
  for( i = 0; i < * pSize; i ++ ) (* ppMem)[ i ] = 0.5;
  for ( i=0; i< nrow-1; i++ )
  {
    for ( j=0;j<= ncol; j++ )
    {
        printf("%d\t Image: %f\n",i*nrow + j, image[0][i*nrow + j]);
        image[0][i*nrow + j]=1.5;
    }
  }

}