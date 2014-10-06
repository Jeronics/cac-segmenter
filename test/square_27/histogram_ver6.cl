#define SQRT_2PI     2.5066282746310

// global size (0) == stride 
// global size (1) == nintervals / 4

// local size (0) = 32 
// local size (1) = 8 (per cada bloc de 32 nivells de grisos)

__kernel void histo_energy_thread(
    __global float *hist,
    __global float *gray,
    __global float *interp_u,                                                
    __global int *label,                                                   
    int ninterval,
    int npoints,  
    float sigma2)                                             
{
  int lid0   = get_local_id(0);
  int gid0   = get_global_id(0); 
  int stride = get_global_size(0);

  int lid1   = get_local_id(1);
  int pos1   = get_group_id(1) * 32;

  int   i;
  int   iter  = ceil((float) npoints / (float) stride);
  float sigma = sqrt(sigma2);

  __local float Ip[32], graylocal[32], hist_thread[32*32];
  __local int lab[32];

  for(int k = lid1; k < 32; k += 8)
    hist_thread[k * 32 + lid0] = 0.0;

  if (lid1 == 0)
    graylocal[lid0] = gray[pos1 + lid0];

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int it = 0; it < iter; it++)
  {
    i = gid0 + stride * it; 

    if ((lid1 == 0) && (i < npoints)) {
      Ip[lid0]  = interp_u[i];
      lab[lid0] = label[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((lab[lid0]) && (i < npoints)) {
      for(int k = lid1; k < 32; k += 8) 
	hist_thread[k * 32 + lid0] += exp(-(pown(graylocal[k] - Ip[lid0],2)/(2.0 * sigma2))); 
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for(int k = lid1; k < 32; k += 8) {
    float val = hist_thread[k * 32 + lid0] / (SQRT_2PI * sigma);
    hist[stride * (pos1 + k) + gid0] = val;
  }

  //barrier(CLK_GLOBAL_MEM_FENCE);
}

// global size(0) == number of intervals 

__kernel void histo_energy_reduce(
    __global float *p,
    __global float *hist,
    int stride)
{
  int gid0 = get_global_id(0);
  int j    = gid0 * stride; 

  float bin = 0.0;

  for(int i = 0; i < stride; i++)
    bin += hist[i + j];

 // printf("gid0 = %d %f\n", gid0, bin);
  p[gid0] = bin;
}

// (0) == i, global_size(0) = stride
// (1) == vertex, global_size(1) = nvertexs

// workgroup size (0) = 32, 
// workgroup size (1) = 4 vertex 

__kernel void histo_gradient_thread(
    __global float *deriv_x,
    __global float *deriv_y,
    __global float *gray,
    __global float *weight,
    __global float *interp_u,
    __global float *interp_ux,
    __global float *interp_uy,
    __global float *affine_coordinates,
    __global int *label,
    int ninterval,
    int npoints,
    float sigma2)
{
  int lid0   = get_local_id(0);      // 0...31
  int gid0   = get_global_id(0);     // i
  int stride = get_global_size(0);

  int lid1    = get_local_id(1);     // 0...3
  int gid1    = get_global_id(1);    // gid1 == vertex
  int nvertex = get_global_size(1);

  int i;
  int iter     = ceil((float) npoints / (float) stride);
  float sigma  = sqrt(sigma2);

  float sum_x, sum_y, affcoor;

  __local int lab[32];
  __local float Ip[32], Ipx[32], Ipy[32];
  __local float graylocal[256], w[256]; 
  __local float dx[32*4], dy[32*4];

  if (lid1 == 0)
  {
    for(int k = lid0; k < ninterval; k += 32)
    {
      graylocal[k] = gray[k]; 
      w[k] = weight[k];
    }
  }

  dx[lid1 * 32 + lid0] = 0.0;
  dy[lid1 * 32 + lid0] = 0.0;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int it = 0; it < iter; it++)
  {
    i = gid0 + stride * it;

    if (i < npoints)
    {
      if (lid1 == 0) 
      {
	lab[lid0] = label[i];
	Ip[lid0]  = interp_u[i];
	Ipx[lid0] = interp_ux[i];
	Ipy[lid0] = interp_uy[i];
      }

      affcoor = affine_coordinates[gid1 * npoints + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((lab[lid0]) && (i < npoints))
    {
      sum_x = 0.0;
      sum_y = 0.0;

      for(int k = 0; k < ninterval; k++)
      {
	float x = graylocal[k] - Ip[lid0];
	float prod = w[k] * x * exp(-pown(x,2)/(2.0 * sigma2));
	float prod_x = prod * Ipx[lid0];
	float prod_y = prod * Ipy[lid0];

	sum_x += prod_x * affcoor;
	sum_y += prod_y * affcoor;
      }

      dx[lid1 * 32 + lid0] += sum_x;
      dy[lid1 * 32 + lid0] += sum_y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float den = SQRT_2PI * pown(sigma,3);
  deriv_x[gid1 * stride + gid0] = dx[lid1 * 32 + lid0] / den;
  deriv_y[gid1 * stride + gid0] = dy[lid1 * 32 + lid0] / den;
}

// global size(0) == nvertexs

__kernel void histo_gradient_reduce(
    __global float *d_x,
    __global float *d_y,
    __global float *deriv_x,
    __global float *deriv_y,
    int stride)
{
  int i, idx;
  int gid0    = get_global_id(0);        // vertex
  int j       = stride * gid0;

  float x = 0.0;
  float y = 0.0;

  for(i = 0; i < stride; i++)
  {
    idx = i + j;
    x += deriv_x[idx];
    y += deriv_y[idx];
  }

  d_x[gid0] = x;
  d_y[gid0] = y;
}

