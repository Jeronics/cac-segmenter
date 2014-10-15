/* api_image.c */
struct image *cac_image_new(void);
struct image *cac_image_alloc(int ncol, int nrow);
void cac_image_delete(struct image *image);
void cac_image_copy(struct image *out, struct image *in);
void cac_image_clear(struct image *im, double value);
