#include "api_cac.h"

void cac_get_omega1_omega2(
     int *omega1_size,          /* output */
     double **omega1_coord,     /* output */
     int *omega2_size,          /* output */
     double **omega2_coord,     /* output */
     int contour_size,          /* input */
     double *contour_coord,     /* input */
     int ncol,                  /* input */
     int nrow,                  /* input */
     int band_size)             /* input */
{
  struct image *mask;
  struct image *dist_int, *dist_ext;

  int k, tmp1, tmp2;

  /* Draw contour on an image and fill is interior */

  mask = cac_image_alloc(ncol, nrow);
  cac_image_clear(mask, 0.0);

  tmp2 = contour_size - 1; 

  for(k = 0; k < tmp2; k++)
  {
    tmp1 = k + 1;
    cac_image_line_draw(mask, round(contour_coord[2*k]), round(contour_coord[2*k+1]), 
                              round(contour_coord[2*tmp1]), round(contour_coord[2*tmp1+1]), 1.0);
  }

  tmp1 = 0;
  cac_image_line_draw(mask, round(contour_coord[2*k]), round(contour_coord[2*k+1]), 
                            round(contour_coord[2*tmp1]), round(contour_coord[2*tmp1+1]), 1.0);

  /* Fill the interior of the contour */

  cac_holefilling(mask);

  /* Compute distance function for the exterior pixels limited to the band size */

  dist_ext = cac_image_alloc(ncol, nrow);
  cac_fdistance(dist_ext, mask, band_size);
  cac_get_pixels_omega(omega2_size, omega2_coord, mask, dist_ext);

  /* Create an inverted copy of the mask we have obtained previously */

  cac_mask_invert(mask);

  dist_int = cac_image_alloc(ncol, nrow);
  cac_fdistance(dist_int, mask, band_size);
  cac_get_pixels_omega(omega1_size, omega1_coord, mask, dist_int);

  /* Done. We may delete non  necessary memory */
  cac_image_delete(mask);
  cac_image_delete(dist_ext);
  cac_image_delete(dist_int);
}

/**
 *
 * Draw a line on an image with gray level c between (a0,b0) and (a1,b1) 
 *
 */

void cac_image_line_draw(struct image *image, int a0, int b0, int a1, int b1, float c)
{ 
  int bdx,bdy;
  int sx,sy,dx,dy,x,y,z;

  if ((!image) || (!image->gray)) 
    cac_error("NULL image struct or NULL gray plane\n");

  bdx = image->ncol;
  bdy = image->nrow;

  if (a0<0) a0=0; else if (a0>=bdx) a0=bdx-1;
  if (a1<0) 
  { 
    a1=0; 
  }
  else 
    if (a1>=bdx) 
    {
      a1=bdx-1; 
    }
  if (b0<0) b0=0; else if (b0>=bdy) b0=bdy-1;
  if (b1<0) 
  { 
    b1=0; 
  }
  else if (b1>=bdy) 
  { b1=bdy-1; 
  }

  if (a0<a1) { sx = 1; dx = a1-a0; } else { sx = -1; dx = a0-a1; }
  if (b0<b1) { sy = 1; dy = b1-b0; } else { sy = -1; dy = b0-b1; }
  x=0; y=0;

  if (dx>=dy) 
  {
    z = (-dx) / 2;
    while (abs(x) <= dx) 
    {
      image->gray[(y+b0)*bdx+x+a0] = c;
      x+=sx;
      z+=dy;
      if (z>0) { y+=sy; z-=dx; }
    } 
  }
  else 
  {
    z = (-dy) / 2;
    while (abs(y) <= dy) {
      image->gray[(y+b0)*bdx+x+a0] = c;
      y+=sy;
      z+=dx;
      if (z>0) { x+=sx; z-=dy; }
    }
  }
}

/*
 * 
 *  Compute distance function limited to a certain band
 *
 */

#define TMP(i,j) (ptrTMP[(j)*ncIN + (i)])

void cac_fdistance(
    struct image *out,
    struct image *in, 
    int N)
{
  struct image *tmp;

  float a;
  int n, changed;
  int i,j,ncIN;
  float *ptrTMP, *ptrOUT;

  tmp = cac_image_alloc(in->ncol, in->nrow);

  cac_image_copy(tmp, in);
  cac_image_copy(out, in);

  ncIN = in->ncol;

  for (n = 1; n <= N; n++)
  {
    if (n > 1) cac_image_copy(tmp,out);

    ptrTMP = tmp->gray;  
    ptrOUT = out->gray;

    changed = 0;

    for(j=0;j<(in->nrow);j++)
      for(i=0;i<(in->ncol);i++, ptrOUT++)
      {
	if (*ptrOUT != 0.0)
	  continue;

	a = TMP(i,j);

	if (j==0 && i==0)   
	{
	  a = MAX(a,TMP(i+1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i+1,j+1));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i+1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i+1,j+1));
	}
	else  if (j==0 && i==((ncIN)-1))    
	{
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i-1,j+1));
	}
	else   if (j==0 && i!=0 && i!=((ncIN)-1))    
	{
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i+1,j+1));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i+1,j+1));
	}
	else  if (j==((in->nrow)-1) && i==0)    
	{
	  a = MAX(a,TMP(i+1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i+1,j-1));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i+1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i+1,j-1));
	}
	else if (j==((in->nrow)-1) && i==((ncIN)-1))    
	{
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i-1,j-1));
	}
	else if (j==((in->nrow)-1) && i!=0 && i!=((ncIN)-1))    
	{
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i+1,j-1));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i+1,j-1));
	}
	else  if (i==0 && j!=0 && j!=((in->nrow)-1))    
	{
	  a = MAX(a,TMP(i+1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i+1,j+1));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i+1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i+1,j-1));
	}
	else  if (i==((ncIN)-1) && j!=0 && j!=((in->nrow)-1))    
	{
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i-1,j-1));
	}
	else {
	  a = MAX(a,TMP(i-1,j-1));
	  a = MAX(a,TMP(i,j-1));
	  a = MAX(a,TMP(i+1,j-1));
	  a = MAX(a,TMP(i-1,j));
	  a = MAX(a,TMP(i+1,j));
	  a = MAX(a,TMP(i-1,j+1));
	  a = MAX(a,TMP(i,j+1));
	  a = MAX(a,TMP(i+1,j+1));
	}

	if (a == 0)
	  continue;

	if (n == 1)
	  *ptrOUT = a;
	else
	  *ptrOUT = a + 1;

	changed = 1;
      }

    if (!changed) break;
  }

  ptrTMP = in->gray;  
  ptrOUT = out->gray;

  changed = 0;

  for(j=0;j<(in->nrow);j++)
    for(i=0;i<(in->ncol);i++, ptrOUT++, ptrTMP++)
      if (*ptrTMP)
	*ptrOUT = 0.0;

  cac_image_delete(tmp);
}

#undef TMP

/*
 *
 * Given a mask image, invert it
 *
 */

void cac_mask_invert(struct image *inout)
{
  int i, size;
  float *p;

  size = inout->nrow * inout->ncol;

  p  = inout->gray;

  for(i = 0; i < size; i++, p++)
  {
    if (*p)
      *p = 0;
    else
      *p = 1.0;
  }
}

/*
 * 
 * Given an mask image, return the number of pixels that belong to the set
 * omega. Note that the condition for a pixel p belonging to omega is: pixel p
 * should have value mask[p] == 0 and dist[p] != 0. 
 *
 */

int cac_get_number_pixels_omega(struct image *mask, struct image *dist)
{
  int count;
  float *p_mask, *p_dist, *p_mask_end;

  count = 0;

  p_dist = dist->gray;

  p_mask     = mask->gray;
  p_mask_end = p_mask + mask->nrow * mask->ncol;

  for(; p_mask < p_mask_end; p_mask++, p_dist++)
    if ((*p_mask == 0) && (*p_dist != 0)) 
      count++; 

  if (count == 0)
    cac_error("cac_get_number_pixels_omega: count == 0, this should not happen\n"); 

  return count;
}

/*
 * 
 * Given an mask image, return a vector with the coordinates of the
 * pixels that belong to the mask. Used to get the interior and 
 * exterior pixels during the cage active contour evolutio.
 *
 */

void cac_get_pixels_omega(int *omega_size, double **omega_coord, struct image *mask, struct image *dist)
{
  float *p_mask, *p_dist;
  double *coord;
  int n;

  n = cac_get_number_pixels_omega(mask, dist);

  coord = cac_xmalloc(sizeof(double) * n * 2);

  p_mask = mask->gray;
  p_dist = dist->gray;

  int i = 0;
  int row, col;

  for(row = 0; row < mask->nrow; row++) 
  {
    for(col = 0; col < mask->ncol; col++, p_mask++, p_dist++)
    {
      if ((*p_mask == 0) && (*p_dist != 0))
      {
	coord[2*i]   = col;
	coord[2*i+1] = row;
	i++;
      }
    }
  }

  if (i != n) 
    cac_error("error: get_pixels_omega, i != n"); 

  *omega_size = n;
  *omega_coord = coord;
}

/*
 *
 * The hole filling algorithm
 *
 */

void cac_holefilling(struct image *mask)
{
  int i, j;

  /* The algorithm is not "perfect": it begins propagation assuming that 
     the point at (0,0) belongs to the background */
  i = 0;
  j = 0;

  /* Check if value is 0 at this point */
  if (mask->gray[i * mask->ncol + j] != 0)
    cac_error("ERROR: cac_holefilling (0,0) point is not zero\n");

  /* Perform holefilling */
  cac_holefilling_child(mask, j, i);

  /* Invert. Thus the interior of the region is painted */
  cac_mask_invert(mask);
}

/*
 *
 * Hole filling algorithm. Used to compute interior of a region. Do not
 * call directly, use cac_holefilling instead.
 *
 */

void cac_holefilling_child(struct image *mask, int x, int y)
{
  struct queue *q;
  struct point point;

  int k;
  int xp, yp;
  int new_xp, new_yp, new_offset;

  int ncol = mask->ncol;
  int nrow = mask->nrow;

  int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};

  q = cac_new_queue(sizeof(struct point), 4 * (nrow + ncol));

  point.x = x;
  point.y = y;

  mask->gray[y * ncol + x] = 1.0;
  cac_put_elem_queue((void *) &point, q);

  while (!cac_is_queue_empty(q))
  {
    cac_get_elem_queue((char *) &point, q);

    xp = point.x;
    yp = point.y;

    for(k = 0; k < 4; k++)
    {
      new_xp = xp + dx[k];
      new_yp = yp + dy[k];

      if ((new_xp < 0) || (new_yp < 0) || (new_xp >= mask->ncol)  || (new_yp >= mask->nrow))
	continue;

      new_offset = new_yp * ncol + new_xp;

      if (mask->gray[new_offset] == 0)
      {
	mask->gray[new_offset] = 1.0;

	point.x = new_xp;
	point.y = new_yp;

	cac_put_elem_queue((char *) &point, q);
      }
    }
  }

  cac_delete_queue(q);
}



/*
 *
 * The next functions are used to manage a queue. Used for 
 * the hole filling algorithm.
 *
 */

struct queue *cac_new_queue(int size_elem, int expand_size)
{ 
  int size;
  struct queue *q;

  size = sizeof(char) * size_elem * expand_size;

  q = (struct queue *) cac_xmalloc(sizeof(struct queue)); 

  q->b = (char *) cac_xmalloc(size);
  q->e = q->b + size; 

  q->r = q->w = q->b;

  q->size_elem   = size_elem;
  q->nb_elem     = 0;
  q->expand_size = expand_size;
  q->max_elem    = expand_size;

  return q;
}

void cac_delete_queue(q)
  struct queue *q;
{
  if (!q)
    cac_error("[cac_delete_queue]: queue is not allocated.\n");

  free((char *) q->b);
  free(q);
}


int cac_is_queue_empty(q)
  struct queue *q;
{
  int rt;

  if (!q)
    cac_error("[cac_is_queue_empty]: queue is not allocated.\n");

  rt = 0;

  if (!q->nb_elem)
    rt = 1;

  return rt;
}

int cac_is_queue_full(q)
  struct queue *q;
{
  int rt;

  if (!q)
    cac_error("[cac_is_queue_full]: queue is not allocated.\n");

  rt = 0;

  if (q->nb_elem == q->max_elem)
    rt = 1;

  return rt;

}

void cac_put_elem_queue(elem, q)
  char *elem;
  struct queue *q;
{
  if (!q)
    cac_error("[cac_put_queue]: queue is not allocated.\n");

  if (cac_is_queue_full(q))
    cac_expand_queue(q);

  /* Copy data into the queue */

  memcpy(q->w, elem, sizeof(char) * q->size_elem);

  /* Increment write pointer and number of elements */

  q->nb_elem++;
  q->w += sizeof(char) * q->size_elem;

  /* Check if we are out of range in the queue */

  if (q->w == q->e)
    q->w = q->b;
}


void cac_get_elem_queue(elem, q)
  char *elem; 
  struct queue *q;
{
  if (!q)
    cac_error("[cac_get_queue]: queue is not allocated.\n");

  if (cac_is_queue_empty(q))
    cac_error("[cac_get_queue]: queue is empty, no elements to extact.\n");

  /* Copy from queue to data */

  memcpy(elem, q->r, sizeof(char) * q->size_elem);

  /* Decrement number of elements, increment read pointer */

  q->nb_elem--;
  q->r += sizeof(char) * q->size_elem;

  /* Check if we are out of range in the queue */

  if (q->r == q->e)
    q->r = q->b;
}

int cac_get_queue_nb_elements(q)
  struct queue *q;
{
  if (!q)
    cac_error("[cac_get_queue_nb_elements]: queue is not allocated.\n");

  return (q->nb_elem);
}

void cac_expand_queue(q)
  struct queue *q;
{
  char *elem;
  struct queue *new_q;

  /* Create new queue */

  new_q = cac_new_queue(q->size_elem, q->max_elem + q->expand_size);

  /* Copy contents of the current queue to the new one */

  elem = (char *) cac_xmalloc(q->size_elem);

  while (!cac_is_queue_empty(q))
  {
    cac_get_elem_queue(elem, q);
    cac_put_elem_queue(elem, new_q);
  }

  free(elem);

  /* Now reassign pointers and free memory */

  free(q->b);

  q->b = new_q->b;
  q->e = new_q->e;
  q->r = new_q->r;
  q->w = new_q->w;

  q->max_elem = new_q->max_elem;
}



