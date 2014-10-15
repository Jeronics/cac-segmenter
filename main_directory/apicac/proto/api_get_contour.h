/* api_get_contour.c */
void cac_contour_get_interior_contour(int *contour_size, double **contour_coord, double *img, int ncol, int nrow, int conn);
void cac_contour_blur(double *contour_coord, int contour_size);
void cac_circular_convolution(double *output, double *input, int npoints, double *filter_coefs, int filter_length);
void cac_gaussian_filter(double sigma, double ratio, int max_samples, double **filter, int *nsamples);
