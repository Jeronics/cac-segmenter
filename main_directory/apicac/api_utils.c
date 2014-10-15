#include "api_cac.h"

/**
 *
 * Print error message
 *
 */

void cac_error(char *str)
{
  fprintf(stderr, "%s\n", str);
  exit(EXIT_FAILURE);
}


/**
 *
 * Malloc with error checking
 *
 */

void *cac_xmalloc(size_t size)
{
  void *p = malloc(size);
  if (!p) {
    printf("xmalloc: out of memory\n");
    exit(1);
  }

  return p;
}



