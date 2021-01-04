// Author: Ayush Kumar
// Roll No: 170195
// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

typedef unsigned long long int uint64_cu;

const uint64_cu N = (1 << 12);
#define BLOCK_SIZE 32
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(uint64_cu* d_A, uint64_cu* d_B, uint64_cu* d_C) {
  // TODO: Fill in
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < N; k++) {
    d_C[i*N + j] += d_A[i*N + k] * d_B[k*N + j];
  }
}

__global__ void kernel2(uint64_cu* d_A, uint64_cu* d_B, uint64_cu* d_C) {
  // TODO: Fill in
  // Block row and column
  int block_i = blockIdx.y;
  int block_j = blockIdx.x;

  // Each thread block computes one sub-matrix B_sub of B
  uint64_cu* C_sub = &d_C[block_i * blockDim.y * N + block_j * blockDim.x];

  // Each thread computes one element of B_sub
  // by accumulating results into B_value
  uint64_cu C_value = 0;

  // Thread row and column within B_sub
  int thread_i = threadIdx.y;
  int thread_j = threadIdx.x;

  // Loop over all the sub-matrices of A^T and A that are
  // required to compute B_sub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < N/blockDim.x; m++) {
    // Get sub-matrix AT_sub of AT
    uint64_cu* A_sub = &d_A[block_i * blockDim.y * N + m * blockDim.x];
    // Get sub-matrix A_sub of A
    uint64_cu* B_sub = &d_B[m * blockDim.y * N + block_j * blockDim.x];

    // Shared memory used to store AT_sub and A_sub respectively
    __shared__ uint64_cu As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ uint64_cu Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load AT_sub and A_sub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[thread_i][thread_j] = A_sub[thread_i * N + thread_j];
    Bs[thread_i][thread_j] = B_sub[thread_i * N + thread_j];

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply AT_sub and A_sub together
    for (int k = 0; k < BLOCK_SIZE; k += 16) {
      C_value += As[thread_i][k] * Bs[k][thread_j];
      C_value += As[thread_i][k+1] * Bs[k+1][thread_j];
      C_value += As[thread_i][k+2] * Bs[k+2][thread_j];
      C_value += As[thread_i][k+3] * Bs[k+3][thread_j];
      C_value += As[thread_i][k+4] * Bs[k+4][thread_j];
      C_value += As[thread_i][k+5] * Bs[k+5][thread_j];
      C_value += As[thread_i][k+6] * Bs[k+6][thread_j];
      C_value += As[thread_i][k+7] * Bs[k+7][thread_j];
      C_value += As[thread_i][k+8] * Bs[k+8][thread_j];
      C_value += As[thread_i][k+9] * Bs[k+9][thread_j];
      C_value += As[thread_i][k+10] * Bs[k+10][thread_j];
      C_value += As[thread_i][k+11] * Bs[k+11][thread_j];
      C_value += As[thread_i][k+12] * Bs[k+12][thread_j];
      C_value += As[thread_i][k+13] * Bs[k+13][thread_j];
      C_value += As[thread_i][k+14] * Bs[k+14][thread_j];
      C_value += As[thread_i][k+15] * Bs[k+15][thread_j];
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of AT and A in the next iteration
    __syncthreads();
  }
  // Write B_sub to device memory
  // Each thread writes one element
  C_sub[thread_i * N + thread_j] = C_value;
}

__host__ void cpumatMul(uint64_cu* h_A, uint64_cu* h_B, uint64_cu* h_C) {
  for (uint64_cu i = 0; i < N; i++) {
    for (uint64_cu j = 0; j < N; j++) {
      float sum = 0.0;
      for (uint64_cu k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(uint64_cu* w_ref, uint64_cu* w_opt) {
  bool wrong = false;
  for (uint64_cu i = 0; i < N; i++) {
    for (uint64_cu j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
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
  uint64_cu SIZE = N * N;

  uint64_cu *h_A, *h_B, *h_cpu_C, *h_gpu1_C, *h_gpu2_C;

  h_A = (uint64_cu*)malloc(SIZE * sizeof(uint64_cu));
  h_B = (uint64_cu*)malloc(SIZE * sizeof(uint64_cu));
  h_cpu_C = (uint64_cu*)malloc(SIZE * sizeof(uint64_cu));
  h_gpu1_C = (uint64_cu*)malloc(SIZE * sizeof(uint64_cu));
  h_gpu2_C = (uint64_cu*)malloc(SIZE * sizeof(uint64_cu));

  for (uint64_cu i = 0; i < N; i++) {
    for (uint64_cu j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 64;
      h_B[i * N + j] = 2;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
      h_gpu2_C[i * N + j] = 0;
    }
  }

  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float kernel_time;

  uint64_cu *d_A, *d_B, *d_C1;
  // // if (status != cudaSuccess) {
  // //   cerr << cudaGetErrorString(status) << endl;
  // // }
  status = cudaMalloc(&d_A, SIZE * sizeof(uint64_cu));
  status = cudaMalloc(&d_B, SIZE * sizeof(uint64_cu));
  status = cudaMalloc(&d_C1, SIZE * sizeof(uint64_cu));
  dim3 threads_in_block1(32, 32);
  dim3 blocks_in_grid1(N/threads_in_block1.x, N/threads_in_block1.y);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_cu), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_cu), cudaMemcpyHostToDevice);
  status = cudaMemset(d_C1, 0, SIZE * sizeof(uint64_cu));
  // TODO: Fill in
  kernel1<<<blocks_in_grid1, threads_in_block1>>>(d_A, d_B, d_C1);
  cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_cu), cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_cpu_C, h_gpu1_C);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  uint64_cu* d_C2;
  status = cudaMalloc(&d_C2, SIZE * sizeof(uint64_cu));
  dim3 threads_in_block2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_in_grid2(N/threads_in_block2.x, N/threads_in_block2.y);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_cu), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_cu), cudaMemcpyHostToDevice);
  status = cudaMemset(d_C2, 0, SIZE * sizeof(uint64_cu));
  // TODO: Fill in
  kernel2<<<blocks_in_grid2, threads_in_block2>>>(d_A, d_B, d_C2);
  cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_cu), cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_cpu_C, h_gpu2_C);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  // cudaFree(d_C1);
  cudaFree(d_C2);

  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);
  free(h_gpu2_C);

  return EXIT_SUCCESS;
}
