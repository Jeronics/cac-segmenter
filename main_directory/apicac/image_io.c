// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2013, Lluis Garrido <lluis.garrido@ub.edu>
// All rights reserved.

#include "api_cac.h"

#include <ctype.h>

#define FALSE 0
#define TRUE  1

/**
 *
 * Read and image from disc. Returns allocated structure. 
 *
 */

struct image *cac_pgm_read_image(char * name)
{
  FILE *f;
  int c,items,bin,x,y;
  int xsize,ysize,depth;
  struct image *image;

  bin = FALSE;

  /* open file */
  f = fopen(name,"r");
  if( f == NULL ) {
    fprintf(stderr, "Can't open input file %s.", name);
    exit(1);
  }

  /* read header */
  if( getc(f) != 'P' ) {
    fprintf(stderr, "File %s is not a PGM file!\n", name);
    exit(1);
  }

  c = getc(f);
  if (c == '2')	
    bin = FALSE;
  else if( c == '5' ) 
    bin = TRUE;
  else {
    fprintf(stderr, "The PGM file %s is not supported (requires P2 or P5 header)\n", name);
    exit(1);
  }

  c = getc(f);
  while (isspace(c)) c = getc(f);     /* skip to end of line*/

  while (c == '#')                    /* skip comment lines */
  {
    c = getc(f);
    while (c != '\n') c = getc(f);    /* skip to end of comment line */
    c = getc(f);
  }

  fseek(f, -1, SEEK_CUR);             /* backup one character*/

  fscanf(f,"%d",&xsize);
  fscanf(f,"%d",&ysize);
  fscanf(f,"%d",&depth);

  c = fgetc(f); /*skip the end of line*/

  /* get memory */
  image = cac_image_alloc(xsize, ysize);

  /* read data */
  if (bin == FALSE) {
    for(y=0;y<ysize;y++)
      for(x=0;x<xsize;x++)
      {
	items = fscanf(f, "%d", &c);
	if (items == 0) {
	  fprintf(stderr, "ERROR reading PGM file %s. Arrived to end before expected.\n", name);
	  exit(1);
	}
	image->gray[ x + y * xsize ] = c;
      }
  }
  else
  {
    for(y=0;y<ysize;y++)
      for(x=0;x<xsize;x++)
      {
	c = fgetc(f);
	if (feof(f)) {
	  fprintf(stderr, "ERROR reading PGM file %s. Arrived to end before expected.\n", name);
	  exit(1);
	}
	image->gray[ x + y * xsize ] = c;
      }
  }

  /* close file */
  fclose(f);

  return image;
}

/**
 *
 * Write an image to disc. 
 *
 */

void cac_pgm_write_image(struct image *image, char * name)
{
  FILE * f;
  double data;
  int x,y,n;

  /* open file */
  f = fopen(name,"w");
  if( f == NULL ) {
    fprintf(stderr, "Can't open output file %s.", name);
    exit(1);
  }

  /* write headder */
  fprintf(f,"P2\n");
  fprintf(f,"%d %d\n",image->ncol,image->nrow);
  fprintf(f,"255\n");

  /* write data */
  for(n=1,y=0; y<image->nrow; y++)
    for(x=0; x<image->ncol; x++,n++)
    {
      data = image->gray[ x + y * image->ncol ];
      if (data < 0) data = 0;
      if (data > 255 ) data = 255;
      fprintf(f,"%d ",(int)data);
      if(n==16)
      {
	fprintf(f,"\n");
	n = 0;
      }
    }

  /* close file */
  fclose(f);
}

/**
 *
 * Write an image to disc. 
 *
 */

void cac_write_raw_image(struct image *image, char * name)
{
  int size;
  FILE *fp;

  fp = fopen(name, "w");
  if (!fp)
    cac_error("Could not open file for raw image write.\n");

  size = image->nrow * image->ncol;
  fwrite(image->gray, size, sizeof(float), fp);

  fclose(fp);
}

