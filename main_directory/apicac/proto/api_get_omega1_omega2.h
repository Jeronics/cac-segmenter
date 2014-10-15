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
