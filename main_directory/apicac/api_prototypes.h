/* api_get_affcoord.c */
void cac_get_affine_coordinates(double *affcoord, int set_size, double *set_coord, int ctrlpoints_size, double *ctrlpoints_coord);
void cac_get_affine_coordinates_pixel(double *affcoord, double coord_x, double coord_y, int ctrlpoints_size, double *ctrlpoints_coord);
/* api_get_contour.c */
void cac_contour_get_interior_contour(int *contour_size, double **contour_coord, double *img, int ncol, int nrow, int conn);
void cac_contour_blur(double *contour_coord, int contour_size);
void cac_circular_convolution(double *output, double *input, int npoints, double *filter_coefs, int filter_length);
void cac_gaussian_filter(double sigma, double ratio, int max_samples, double **filter, int *nsamples);
/* api_get_omega1_omega2.c */
void cac_get_omega1_omega2(int *omega1_size, double **omega1_coord, int *omega2_size, double **omega2_coord, int contour_size, double *contour_coord, int ncol, int nrow, int band_size);
void cac_image_line_draw(struct image *image, int a0, int b0, int a1, int b1, float c);
void cac_fdistance(struct image *out, struct image *in, int N);
void cac_mask_invert(struct image *inout);
int cac_get_number_pixels_omega(struct image *mask, struct image *dist);
void cac_get_pixels_omega(int *omega_size, double **omega_coord, struct image *mask, struct image *dist);
void cac_holefilling(struct image *mask);
void cac_holefilling_child(struct image *mask, int x, int y);
struct queue *cac_new_queue(int size_elem, int expand_size);
void cac_delete_queue(struct queue *q);
int cac_is_queue_empty(struct queue *q);
int cac_is_queue_full(struct queue *q);
void cac_put_elem_queue(char *elem, struct queue *q);
void cac_get_elem_queue(char *elem, struct queue *q);
int cac_get_queue_nb_elements(struct queue *q);
void cac_expand_queue(struct queue *q);
/* api_image.c */
struct image *cac_image_new(void);
struct image *cac_image_alloc(int ncol, int nrow);
void cac_image_delete(struct image *image);
void cac_image_copy(struct image *out, struct image *in);
void cac_image_clear(struct image *im, double value);
/* api_utils.c */
void cac_error(char *str);
void *cac_xmalloc(size_t size);
/* image_io.c */
struct image *cac_pgm_read_image(char *name);
void cac_pgm_write_image(struct image *image, char *name);
void cac_write_raw_image(struct image *image, char *name);
