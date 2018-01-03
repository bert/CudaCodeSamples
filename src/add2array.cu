/*!
 * \file add2array.cu
 *
 * \brief Add two arrays of float.
 *
 * Example code from https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
 *
 * Compile it with nvcc, the CUDA C++ compiler:
 *
 *    nvcc add2array.cu -o add2array
 *
 * Run with:
 *    ./add2array
 *
 * Profile with:
 *
 *    nvprof ./add2array
 */


#include <iostream>
#include <math.h>


/*!
 * \brief Kernel function to add the elements of two arrays (GPU).
 *
 * \param[in] n number of elements.
 * \param[in] x array X.
 * \param[in] y array Y
 */
__global__ void
add (int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}


/*!
 * \brief Main function (CPU).
 */
int
main (void)
{
  int N = 1 << 20;
  float *x, *y;

  /* Allocate Unified Memory – accessible from CPU or GPU. */
  cudaMallocManaged (&x, N * sizeof (float));
  cudaMallocManaged (&y, N * sizeof (float));

  /* Initialize x and y arrays on the host. */
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  /* Run kernel on 1M elements on the GPU. */
  add<<<1, 1>>>(N, x, y);

  /* Wait for GPU to finish before accessing on host. */
  cudaDeviceSynchronize ();

  /* Check for errors (all values should be 3.0f). */
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax (maxError, fabs (y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  /* Free memory. */
  cudaFree (x);
  cudaFree (y);
  
  return 0;
}

/* EOF */
