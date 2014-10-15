#include "api_cac.h"

/** 
 *
 * Get interior contour. Assume image is binary (0 and any other value) 
 *
 */

#define EXPAND_SEQUENCE_CONTOUR  10000

void cac_contour_get_interior_contour(
    int *contour_size,   /* output */
    double **contour_coord, /* output */
    double *img,            /* input */
    int ncol,            /* input */
    int nrow,            /* input */
    int conn)            /* input */
{
  double *values;
  int p, j, k, y, x, delta, npoints;
  int sizimg;

  int cy[8] = { -1,-1, 0, 1, 1, 1, 0,-1};
  int cx[8] = {  0, 1, 1, 1, 0,-1,-1,-1};

  /* Check if connectivity is 4 or 8 ... */

  delta = 0; /* Avoid complaining of compiler */

  if (conn == 4)
    delta = 2;
  else if (conn == 8)
    delta = 1;
  else
    cac_error("ERROR(contour_get_sequence): connectivity must be 4 or 8");

  /* Find first point of contour */

  sizimg = ncol * nrow;

  j = 0;

  while ((img[j] == 0.0) && (j < sizimg)) j++;

  if (j == sizimg)
    cac_error("ERROR(contour_get_sequence): image has no non zero pixel");

  /* Allocate memory for sequence of points of the contour */

  values = cac_xmalloc(sizeof(double) * EXPAND_SEQUENCE_CONTOUR * 2);

  values[0] = (double) (j % ncol); /* x coordinate */
  values[1] = (double) (j / ncol); /* y coordinate */

  /* Seek second contour point */

  k = 2;
  npoints = 0;

  do
  {
    x = values[0] + cx[k]; /* x coordinate */
    y = values[1] + cy[k]; /* y coordinate */

    p = img[y * ncol + x];

    if (p == 0)
    {
      k += delta;
      if (k >= 8) k -= 8;
    }
  }
  while ((p == 0) && (k != 0));

  if (p != 0)  /* Contour may have one pixel */
  {
    values[2] = x; /* x coordinate */
    values[3] = y; /* y coordinate */

    npoints = 1;

    while (1)
    {
      k += 6 - delta;
      if (k >= 8) k -= 8;

      do
      {
	k += delta;
	if (k >= 8) k -= 8;

	x = values[2*npoints]   + cx[k];
	y = values[2*npoints+1] + cy[k];

	p = img[y * ncol + x];
      }
      while (p == 0); 

      if ((x == values[0]) && (y == values[1])) 
      {
	if (conn == 4)
	  if (!((k == 6) && (img[(y + 1) * ncol + x] != 0)))
	    break;

	if (conn == 8)
	  if (!(((k == 6) || (k == 7)) && (img[(y + 1) * ncol + (x - 1)] != 0)))
	    break;
      }

      if (++npoints % EXPAND_SEQUENCE_CONTOUR == 0)
	values = realloc(values, (npoints + EXPAND_SEQUENCE_CONTOUR) * sizeof(double) * 2);

      values[2*npoints]   = x; /* x coordinate */
      values[2*npoints+1] = y; /* y coordinate */
    }
  }

  /* Set output parameters */

  *contour_size  = ++npoints;
  *contour_coord = realloc(values, 2 * npoints * sizeof(double));

  /* Blur contour */

  cac_contour_blur(*contour_coord, *contour_size);
}

/*
 *
 * Blur contour 
 *
 */

void cac_contour_blur(
    double *contour_coord,
    int contour_size)
{
  int i, filter_length, M;
  double *filter_coefs;

  double *x, *x_out, *y, *y_out;

  /* Allocate for input */

  x = (double *) cac_xmalloc(contour_size * sizeof(double));
  y = (double *) cac_xmalloc(contour_size * sizeof(double));

  for(i = 0; i < contour_size; i++)
  {
    x[i] = contour_coord[2*i];
    y[i] = contour_coord[2*i+1];
  }

  /* Allocate for output */

  x_out = (double *) cac_xmalloc(contour_size * sizeof(double));
  y_out = (double *) cac_xmalloc(contour_size * sizeof(double));

  /* Filter */

  cac_gaussian_filter(1.5, 1000, 9, &filter_coefs, &filter_length);

  cac_circular_convolution(x_out, x, contour_size, filter_coefs, filter_length);
  cac_circular_convolution(y_out, y, contour_size, filter_coefs, filter_length);

  /* Copy to output */

  for(i = 0; i < contour_size; i++)
  {
    contour_coord[2*i] = x_out[i];
    contour_coord[2*i+1] = y_out[i];
  }

  /* Free memory */

  free(x);
  free(y);
  free(x_out);
  free(y_out);

  /* TODO. Free filter. Not really nice way to do it. Any 
     better way ? My be changing cac_gaussian_filter and
     cac_circular_convolution: the former should not return the
     displaced filter. */

  M = (filter_length - 1) / 2;
  free(filter_coefs-M);
}

/*
 * 
 * Circular convolution. Used to blur contour
 *
 */

void cac_circular_convolution(
    double *output, 
    double *input, 
    int npoints, 
    double *filter_coefs, 
    int filter_length)
{
  int M;
  double seq, *p_fil_begin, *p_fil, *p_fil_end;
  double *p_out, *p_out_end, *p_begin, *p_in, *p_in_begin, *p_in_end;

  /* Implementation of

     y[m] = SUM(filter_coefs[k] * x[m - k])

   */
  /* Initialize pointers */

  M = (filter_length - 1) / 2;

  /* For m = 0; y[0] = SUM(filter_coefs[k] * x[-k]); and for k = - M;
     we are multiplying filter_coefs[-M] * x[M]. p_begin points to x[M],
     each time m increases p_begin increases also. */

  p_begin  = input + (M % npoints);

  /* Where does the filter_coefs begin ? */

  p_fil_begin = filter_coefs - M;   /* filter_coefs[-M] */
  p_fil_end   = filter_coefs + M;   /* filter_coefs[M]  */

  /* Filter */

  p_out     = output;
  p_out_end = output + npoints;

  p_in_begin = input;
  p_in_end   = input + npoints;

  for(; p_out < p_out_end; p_out++)
  {
    seq   = 0;

    p_in  = p_begin;
    p_fil = p_fil_begin; 

    for(; p_fil <= p_fil_end; p_fil++)
    {
      seq += *p_fil * *p_in;

      if (--p_in < p_in_begin)     /* Circular rotation */
      {
	p_in = p_in_end;
	p_in--;
      }
    }

    *p_out = seq;

    if (++p_begin == p_in_end)
      p_begin = p_in_begin;
  }
}

/*
 *
 * Gaussian filter to filter the input contour
 *
 */

void cac_gaussian_filter(double sigma, double ratio,
    int max_samples, double **filter, int *nsamples)
{
  int i, M;
  double sigma2, value1, value2, *p1, *p2;

  if (!(max_samples & 1))
    max_samples--;

  sigma2    = 2 * sigma * sigma;

  /* Compute how many samples we need. 
Condition: gaussian(value1) >= value0 / ratio */     

  value1 = sqrt(- sigma2 * log(1.0 / ratio));

  /* And the total needed number of samples is ... */

  M         = (int) ceil(value1);
  *nsamples = M * 2 + 1;

  if (*nsamples > max_samples)
  {
    M = (max_samples - 1) >> 1;
    *nsamples = M * 2 + 1;
  }

  /* Allocate memory */

  *filter = (double *) cac_xmalloc(sizeof(double) * *nsamples);
  *filter = *filter + M;

  /* Compute samples */

  p1 = *filter;
  p2 = p1;

  *p1 = 1.0;

  p1++; 
  p2--;

  value2 = 1.0;

  for(i = 1; i <= M; i++, p1++, p2--)
  {
    value1  = exp(- SQR((double) i) / sigma2);
    value2 +=  2.0 * value1;

    *p1 = value1;
    *p2 = value1;
  }

  /* Let's normalize */

  p1 = *filter;
  p2 = p1;

  *p1 /= value2;

  p1++;
  p2--;

  for(i = 1; i <= M; i++, p1++, p2--)
  {
    *p1 /= value2;
    *p2 /= value2;
  } 
}
