// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2013, Lluis Garrido <lluis.garrido@ub.edu>
// All rights reserved.

#include "api_cac.h"

/**
 *
 *  Allocates a new image 
 *
 */

struct image *cac_image_new()
{
  struct image *image;

  image = (struct image *) cac_xmalloc(sizeof(struct image));

  image->nrow = image->ncol = 0;
  image->gray = NULL;

  return (image);
}

/**
 *
 *  Allocates the gray level array  
 *
 */

struct image *cac_image_alloc(
     int ncol,
     int nrow)
{
  int size;
  struct image *image;
  
    image = cac_image_new(); 

  size = nrow * ncol * sizeof(double);
  if (size <= 0)
  {
    fprintf(stderr, "Cannot allocate an image with zero or negative size\n");
    exit(1);
  }

  image->nrow = nrow;
  image->ncol = ncol;
  image->gray = (float *) cac_xmalloc(size);

  return image;
}

/**
 *
 *  Deallocates an image 
 *
 */

void cac_image_delete(
    struct image *image)

{
  if (image == NULL)
  {
    fprintf(stderr, "Cannot delete image: structure is NULL\n");
    exit(1);
  }

  if (image->gray != NULL) 
  {
    free(image->gray);
    image->gray = NULL;
  }
  else
    printf("image->gray is NULL! cannot delete\n");

  free(image);
}

/**
 *
 * Copy an image from in to out 
 *
 */

void cac_image_copy(
    struct image *out, 
    struct image *in) 
{
  if ((!in) || (!out) || (!in->gray) || (!out->gray) 
      || (in->ncol != out->ncol) || (in->nrow != out->nrow)) 
  {
    fprintf(stderr, "Images cannot be copied: they are NULL or images are of different sizes\n");
    exit(1);
  }

  memcpy(out->gray, in->gray, sizeof(float) * in->ncol*in->nrow);
}

/**
 *
 * Initialize an image to a constant value 
 *
 */

void cac_image_clear(
  struct image *im,
  double value)
{
  float *p;
  int i, size;

  if ((!im) || (!im->gray))
    cac_error("Image is not allocated.\n");

  size = im->nrow * im->ncol;
  p    = im->gray;

  for(i = 0; i < size; i++, p++)
    *p = value;
}


