#include "api_cac.h"

/**
 *
 * Compute affine coordinates ordered per point, i.e.
 *  a[i][j] -- affine coordinate for point i, vertex j 
 *
 */

void cac_get_affine_coordinates(
    double *affcoord,            // output, assumed to be allocated at input
    int set_size,                // input inicialment punts de corntorn
    double *set_coord,           // input
    int ctrlpoints_size,         // input
    double *ctrlpoints_coord)    // input
{
  double x, y, *p_affcoord;

  int k;

  // For each point of the list... 
  for(k = 0; k < set_size; k++)
  {
    x = set_coord[2*k];
    y = set_coord[2*k+1];

    p_affcoord = affcoord + k * ctrlpoints_size;
    cac_get_affine_coordinates_pixel(p_affcoord, x, y, ctrlpoints_size, ctrlpoints_coord);
  }
}

/**
 *
 * Get the affine coordinates for a given pixel
 *
 */

void cac_get_affine_coordinates_pixel(double *affcoord, double coord_x, double coord_y, int ctrlpoints_size, double *ctrlpoints_coord)
{
  int j, j_prev, j_next, flag_norm, flag_j;
  double dx1, dy1, dx2, dy2;
  double norm1, norm2;
  double sum_weight;
  double *dd, *a;

  dd = (double*) cac_xmalloc(sizeof(double) * ctrlpoints_size);
  a = (double*)  cac_xmalloc(sizeof(double) * ctrlpoints_size);

  sum_weight = 0.0;

  double sig = 0;
  flag_norm = 0;
  flag_j    = 0;
  for(j = 0; j < ctrlpoints_size; j++)
  {
    j_next = j + 1;
    if (j_next >= ctrlpoints_size)
      j_next = 0;

    dx1   = ctrlpoints_coord[2*j] - coord_x;
    dy1   = ctrlpoints_coord[2*j+1] - coord_y;
    norm1 = sqrt(dx1 * dx1 + dy1 * dy1);

    dx2   = ctrlpoints_coord[2*j_next] - coord_x;
    dy2   = ctrlpoints_coord[2*j_next+1] - coord_y;
    norm2 = sqrt(dx2 * dx2 + dy2 * dy2);

    dd[j] = norm1;
    sig = 1.0;
    if((dx1*dy2-dy1*dx2)<0) sig = -1.0;

    if ((norm1 <= 1e-05) || (norm2 <= 1e-05))
    {
      if (norm1 <= 1e-05) {
	flag_j    = j;
	flag_norm = 1;
	break;
      }
    }
    else
    {
      double val = (dx1*dx2+dy1*dy2)/(norm1*norm2);
      if(val>1) val = 1;
      if(val<-1) val = -1;
      a[j] = sig*tan(0.5*acos(val));

      if (!isfinite(a[j]))
	cac_error("error in computing affine coordinates for a pixel");
    }
  }

  if (flag_norm)
  {
    for(j = 0; j < ctrlpoints_size; j++)
      affcoord[j] = 0.0;

    affcoord[flag_j] = 1.0;
  }
  else
  {
    for(j = 0; j < ctrlpoints_size; j++)
    {
      j_prev = j - 1;
      if (j_prev < 0)
	j_prev = ctrlpoints_size - 1;

      affcoord[j] = (a[j]+a[j_prev])/dd[j];

      sum_weight = sum_weight+affcoord[j];
    }

    for(j = 0; j < ctrlpoints_size; j++)
    {
      affcoord[j] = affcoord[j]/sum_weight;
    }
  }

  free(a);
  free(dd);
}
