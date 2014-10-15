#ifndef CAC_H
#define CAC_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


/*
 * Basic structures needed by the algorithm 
 */

/* The image structure */

struct image {
  int nrow;       /* Number of rows (dy) */
  int ncol;       /* Number of columns (dx) */
  float *gray;    /* The Gray level plane (may be NULL) */
};

/* Queue */
struct queue {  /* Structure for fifo queue */
  int size_elem, nb_elem, expand_size, max_elem;
  char *b,*e; /* Begin and end of the queue */
  char *r, *w; /* Read and write pointers of the queue */
};

/* A point, for use in the queue */
struct point {
  int x;
  int y;
};



/* Other functions */
#define SQR(a)   ((a)*(a))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

/* For gradient computation */
#define D_NONE                     0
#define D_FIRST                    1
#define D_SECOND                   2
#define D_SECOND_MIXED             3

/* Filter types */
#define FILTER_SIMONCELLI          0
#define FILTER_GAUSS               1

/* Filter to apply */
#define FILTER_SIZE                7
#define FILTER_TYPE                0

/* Other includes */
#include "api_prototypes.h"

#endif

