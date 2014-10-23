/* image_io.c */
struct image *cac_pgm_read_image(char *name);
void cac_pgm_write_image(struct image *image, char *name);
void cac_write_raw_image(struct image *image, char *name);
