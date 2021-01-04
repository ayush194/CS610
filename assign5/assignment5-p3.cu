// Author: Ayush Kumar
// Roll No: 170195
// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <iostream>
#include <sys/time.h>

#define SIZE 1024
#define BLOCK_SIZE 16
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

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

__host__ void ATAonCPU(double* M, double* P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i*SIZE + j] += M[k*SIZE + i] * M[k*SIZE + j];
    }
  }
}

__global__ void ATAkernel1(double* A, double* B) {
  // TODO: Fill in
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < SIZE; k++) {
    // atomicAdd(&B[i*SIZE + j], A[k*SIZE + i] * A[k*SIZE + j]);
    B[i*SIZE + j] += A[k*SIZE + i] * A[k*SIZE + j];
  }
}

__global__ void ATAkernel2(double* A, double* B) {
  // TODO: Fill in
  // Block row and column
  int block_i = blockIdx.y;
  int block_j = blockIdx.x;

  // Each thread block computes one sub-matrix B_sub of B
  double* B_sub = &B[block_i * blockDim.y * SIZE + block_j * blockDim.x];

  // Each thread computes one element of B_sub
  // by accumulating results into B_value
  double B_value = 0;

  // Thread row and column within B_sub
  int thread_i = threadIdx.y;
  int thread_j = threadIdx.x;

  // Loop over all the sub-matrices of A^T and A that are
  // required to compute B_sub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < SIZE/blockDim.x; m++) {
    // Get sub-matrix AT_sub of AT
    double* AT_sub = &A[m * blockDim.y * SIZE + block_i * blockDim.x];
    // Get sub-matrix A_sub of A
    double* A_sub = &A[m * blockDim.y * SIZE + block_j * blockDim.x];

    // Shared memory used to store AT_sub and A_sub respectively
    __shared__ double ATs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];

    // Load AT_sub and A_sub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    ATs[thread_i][thread_j] = AT_sub[thread_i * SIZE + thread_j];
    As[thread_i][thread_j] = A_sub[thread_i * SIZE + thread_j];

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply AT_sub and A_sub together
    for (int k = 0; k < BLOCK_SIZE; k += 16) {
        B_value += ATs[k][thread_i] * As[k][thread_j];
        B_value += ATs[k+1][thread_i] * As[k+1][thread_j];
        B_value += ATs[k+2][thread_i] * As[k+2][thread_j];
        B_value += ATs[k+3][thread_i] * As[k+3][thread_j];
        B_value += ATs[k+4][thread_i] * As[k+4][thread_j];
        B_value += ATs[k+5][thread_i] * As[k+5][thread_j];
        B_value += ATs[k+6][thread_i] * As[k+6][thread_j];
        B_value += ATs[k+7][thread_i] * As[k+7][thread_j];
        B_value += ATs[k+8][thread_i] * As[k+8][thread_j];
        B_value += ATs[k+9][thread_i] * As[k+9][thread_j];
        B_value += ATs[k+10][thread_i] * As[k+10][thread_j];
        B_value += ATs[k+11][thread_i] * As[k+11][thread_j];
        B_value += ATs[k+12][thread_i] * As[k+12][thread_j];
        B_value += ATs[k+13][thread_i] * As[k+13][thread_j];
        B_value += ATs[k+14][thread_i] * As[k+14][thread_j];
        B_value += ATs[k+15][thread_i] * As[k+15][thread_j];
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of AT and A in the next iteration
    __syncthreads();
  }
  // Write B_sub to device memory
  // Each thread writes one element
  B_sub[thread_i * SIZE + thread_j] = B_value;
}

__host__ void check_result(double* Test, double* Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      rel_diff = (Test[i*SIZE + j] - Ref[i*SIZE + j]);
      if (fabs(rel_diff) > THRESHOLD) {
        numdiffs++;
        if (rel_diff > maxdiff)
          maxdiff = rel_diff;
      }
    }
  }
  if (numdiffs > 0)
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << " Max Diff = " << maxdiff
         << "\n";
  else
    cout << "No differences found between base and test versions\n";
}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  double* h_in = new double[SIZE*SIZE];
  double* h_cpu_out = new double[SIZE*SIZE];
  double* h_dev_out = new double[SIZE*SIZE];

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      h_in[i*SIZE + j] = i * j * 0.25;
      h_cpu_out[i*SIZE + j] = 0;
      h_dev_out[i*SIZE + j] = 0;
    }
  }

  double clkbegin = rtclock();
  ATAonCPU(h_in, h_cpu_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "A^T.A on CPU: " << ((2.0 * SIZE * SIZE * SIZE) / cpu_time)
       << " GFLOPS; Time = " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  double* d_in;
  double* d_out;
  float kernel_time;
  // TODO: Fill in
  cudaMalloc(&d_in, sizeof(double)*SIZE*SIZE);
  cudaMalloc(&d_out, sizeof(double)*SIZE*SIZE);
  dim3 threads_in_block1(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_in_grid1(SIZE/threads_in_block1.x, SIZE/threads_in_block1.y);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_in, h_in, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(double)*SIZE*SIZE);
  ATAkernel1<<<blocks_in_grid1, threads_in_block1>>>(d_in, d_out);
  cudaMemcpy(h_dev_out, d_out, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_cpu_out, h_dev_out);
  cout << "A^T.A on GPU Kernel 1: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << "CUDA Error: " << cudaGetErrorString(status) << endl;
  }

  dim3 threads_in_block2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_in_grid2(SIZE/threads_in_block2.x, SIZE/threads_in_block2.y);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_in, h_in, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(double)*SIZE*SIZE);
  ATAkernel2<<<blocks_in_grid2, threads_in_block2>>>(d_in, d_out);
  cudaMemcpy(h_dev_out, d_out, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_cpu_out, h_dev_out);
  cout << "A^T.A on GPU Kernel 2: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << "CUDA Error: " << cudaGetErrorString(status) << endl;
  }

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);

  // Free host memory
  delete[] h_in;
  delete[] h_cpu_out;
  delete[] h_dev_out;

  return EXIT_SUCCESS;
}
