#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void first(int *c, int n)
{
    // Get our global thread ID
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (tid < n)
        c[tid] = tid;
    printf(" %d ",tid);
} /*  Printed in order in groups of 32. But other than that, in random order. */
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 2000;
 
    //Host vector
    int *h_c;
 
    //Device output vector
    int *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);
 
    // Allocate memory on host
    h_c = (int*)malloc(bytes);
 
    // Allocate memory on GPU
    cudaCheck(cudaMalloc(&d_c, bytes));
 
    // Copy host vectors to device
    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    first<<<gridSize, blockSize>>>(d_c, n);
 
    // Synchronize
    cudaCheck(cudaDeviceSynchronize());

    // Copy array back to host
    cudaCheck(cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost ));
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    for(int i=0; i<n; i++)
      printf("cpu %d ", h_c[i]);
    printf("\n");
 
    // Release device memory
    cudaFree(d_c);
 
    // Release host memory
    free(h_c);
 
    return 0;
}
