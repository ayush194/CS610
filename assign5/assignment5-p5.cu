// Author: Ayush Kumar
// Roll No: 170195
// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p5.cu -o assignment5-p5

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (256);
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

// TODO: Edit the function definition as required
__global__ void kernel1(float* d_in, float* d_out) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= 1 && j >= 1 && k >= 1 && i < (N-1) && j < (N-1) && k < (N-1)) {
    d_out[i*N*N + j*N + k] = 0.8 * (d_in[(i-1)*N*N + j*N + k] + d_in[(i+1)*N*N + j*N + k] + 
                                    d_in[i*N*N + (j-1)*N + k] + d_in[i*N*N + (j+1)*N + k] + 
                                    d_in[i*N*N + j*N + k-1] + d_in[i*N*N + j*N + k+1]);
  }
}

// TODO: Edit the function definition as required
__global__ void kernel2(float* d_in, float* d_out) {
  // Block row and column
  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int block_k = blockIdx.z;

  // Thread row and column within B_sub
  int thread_i = threadIdx.y;
  int thread_j = threadIdx.x;
  int thread_k = threadIdx.z;
  
  int i = block_i * BLOCK_SIZE_Y + thread_i;
  int j = block_j * BLOCK_SIZE_X + thread_j;
  int k = block_k * BLOCK_SIZE_Z + thread_k;

  // if (i < 1 || i >= N-1 || j < 1 || j >= N-1 || k < 1 || k >= N-1) return;
  if (i >= 1 && j >= 1 && k >= 1 && i < (N-1) && j < (N-1) && k < (N-1)) {
    // Each thread block computes one sub-matrix B_sub of B
    float* d_in_sub = d_in + (block_i*BLOCK_SIZE_Y)*N*N + (block_j*BLOCK_SIZE_X)*N + block_k*BLOCK_SIZE_Z;
    float* d_out_sub = d_out + (block_i*BLOCK_SIZE_Y)*N*N + (block_j*BLOCK_SIZE_X)*N + block_k*BLOCK_SIZE_Z;

    __shared__ float tmp[BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2][BLOCK_SIZE_Z+2];
    // thread_i++; thread_j++; thread_k++;

    // load everything from d_in_sub to tmp
    tmp[1+thread_i][1+thread_j][1+thread_k] = d_in_sub[thread_i*N*N + thread_j*N + thread_k];
    tmp[1+thread_i-1][1+thread_j][1+thread_k] = d_in_sub[(thread_i-1)*N*N + thread_j*N + thread_k];
    tmp[1+thread_i+1][1+thread_j][1+thread_k] = d_in_sub[(thread_i+1)*N*N + thread_j*N + thread_k];
    tmp[1+thread_i][1+thread_j-1][1+thread_k] = d_in_sub[thread_i*N*N + (thread_j-1)*N + thread_k];
    tmp[1+thread_i][1+thread_j+1][1+thread_k] = d_in_sub[thread_i*N*N + (thread_j+1)*N + thread_k];
    tmp[1+thread_i][1+thread_j][1+thread_k-1] = d_in_sub[thread_i*N*N + thread_j*N + thread_k-1];
    tmp[1+thread_i][1+thread_j][1+thread_k+1] = d_in_sub[thread_i*N*N + thread_j*N + thread_k+1];

    __syncthreads();
    float d_out_value =
      0.8 * (tmp[1+thread_i-1][1+thread_j][1+thread_k] + tmp[1+thread_i+1][1+thread_j][1+thread_k] + 
            tmp[1+thread_i][1+thread_j-1][1+thread_k] + tmp[1+thread_i][1+thread_j+1][1+thread_k] + 
            tmp[1+thread_i][1+thread_j][1+thread_k-1] + tmp[1+thread_i][1+thread_j][1+thread_k+1]);
    __syncthreads(); 

    d_out_sub[thread_i*N*N + thread_j*N + thread_k] = d_out_value;
  }
}

// TODO: Edit the function definition as required
__host__ void stencil(float* in, float* out) {
  for (int i=1; i<N -1; i++) {
    for (int j=1; j<N -1; j++) {
      for (int k=1; k<N -1; k++) {
        out[i*N*N + j*N + k] = 0.8 * (in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k] + 
                                      in[i*N*N + (j-1)*N + k] + in[i*N*N + (j+1)*N + k] + 
                                      in[i*N*N + j*N + k-1] + in[i*N*N + j*N + k+1]);
      }
    }
  }
}

__host__ void check_result(float* w_ref, float* w_opt, uint64_t size) {
  bool wrong = false;
  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        if (w_ref[i*N*N + j*N + k] != w_opt[i*N*N + j*N + k]) {
          wrong = true;
          goto out;
        }
      }
    }
  }
out:
  if (wrong) {
    cout << "Diffs found!" << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N * N;

  float* h_in = new float[sizeof(float)*N*N*N];
  float* h_out = new float[sizeof(float)*N*N*N];
  float* h_k1_out = new float[sizeof(float)*N*N*N];
  float* h_k2_out = new float[sizeof(float)*N*N*N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        h_in[i*N*N + j*N + k] = rand() % 10;
      }
    }
  }

  double clkbegin = rtclock();
  stencil(h_in, h_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float kernel_time;

  float *d_in, *d_out;
  cudaMalloc(&d_in, SIZE * sizeof(float));
  cudaMalloc(&d_out, SIZE * sizeof(float));

  // TODO: Fill in kernel1
  // TODO: Adapt check_result() and invoke
  dim3 threads_in_block1(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 blocks_in_grid1(N/threads_in_block1.x, N/threads_in_block1.y, N/threads_in_block1.z);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, SIZE * sizeof(float));
  kernel1<<<blocks_in_grid1, threads_in_block1>>>(d_in, d_out);
  cudaMemcpy(h_k1_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_out, h_k1_out, N);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  // TODO: Fill in kernel2
  // TODO: Adapt check_result() and invoke
  dim3 threads_in_block2(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 blocks_in_grid2(N/threads_in_block2.x, N/threads_in_block2.y, N/threads_in_block2.z);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, SIZE * sizeof(float));
  kernel2<<<blocks_in_grid2, threads_in_block2>>>(d_in, d_out);
  cudaMemcpy(h_k2_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_out, h_k2_out, N);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  // TODO: Free memory
  cudaFree(d_in);
  cudaFree(d_out);

  delete[] h_in;
  delete[] h_out;
  delete[] h_k1_out;
  delete[] h_k2_out;

  return EXIT_SUCCESS;
}
