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
void cac_contour_get_interior_contour(  int* pSize, double **ppMem, double *image, int ncol, int nrow, int conn)
{
  int size;
  double *matriu;
  int i, j;

  // Treballem normalment, sense pensar en els arguments
  // de sortida

  printf("Conectivitat: %d\n", conn);

  for(i = 0; i < nrow; i++)
    for(j = 0; j < ncol; j++)
       printf("Valor de la imatge a [%d][%d]: %f\n", i, j, image[i * ncol + j]);

  size = 2;  // matriu de 4 x 2 (files x columnes, cada fila emmagatzema coordenada x, y del pixel)

  printf("Reservant matriu...\n");
  matriu = malloc(sizeof(double) * size * 2);

  for( i = 0; i < (size * 2); i++) matriu[i] = (double) i;
  for ( i=0; i < 4; i++ )
  {
    for ( j=0; j< 2; j++ )
    {
        printf("Element %d\t de la matriu: %f\n",i*2 + j, matriu[i*2 + j]);
    }
  }

  // Ho preparem per retornar a través dels arguments

  *pSize = size;
  *ppMem = matriu;
}
