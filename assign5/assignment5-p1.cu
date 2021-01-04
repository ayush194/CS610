// Author: Ayush Kumar
// Roll No: 170195
// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p1.cu -o assignment5-p1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 8192
#define SIZE2 8200
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(double* d_k1) {
  // TODO: Fill in
  // int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < SIZE1-1){
    for (int k = 0; k < ITER; k++) {
      for (int i = 1; i < (SIZE1 - 1); i++) {
        d_k1[i*SIZE1 + j+1] = d_k1[(i-1)*SIZE1 + j+1] + d_k1[i*SIZE1 + j+1] + d_k1[(i+1)*SIZE1 + j+1];
      }
    }
  }
}

__global__ void kernel2(double* d_k2) {
  // TODO: Fill in
  // int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int unroll = 8;
  if (j < SIZE2-1) {
    for (int k = 0; k < ITER; k++) {
      int i;
      for (i = 1; i+(unroll-1) < (SIZE2 - 1); i += unroll) {
        d_k2[i*SIZE2 + j+1] = d_k2[(i-1)*SIZE2 + j+1] + d_k2[i*SIZE2 + j+1] + d_k2[(i+1)*SIZE2 + j+1];
        d_k2[(i+1)*SIZE2 + j+1] = d_k2[(i)*SIZE2 + j+1] + d_k2[(i+1)*SIZE2 + j+1] + d_k2[(i+2)*SIZE2 + j+1];
        d_k2[(i+2)*SIZE2 + j+1] = d_k2[(i+1)*SIZE2 + j+1] + d_k2[(i+2)*SIZE2 + j+1] + d_k2[(i+3)*SIZE2 + j+1];
        d_k2[(i+3)*SIZE2 + j+1] = d_k2[(i+2)*SIZE2 + j+1] + d_k2[(i+3)*SIZE2 + j+1] + d_k2[(i+4)*SIZE2 + j+1];
        d_k2[(i+4)*SIZE2 + j+1] = d_k2[(i+3)*SIZE2 + j+1] + d_k2[(i+4)*SIZE2 + j+1] + d_k2[(i+5)*SIZE2 + j+1];
        d_k2[(i+5)*SIZE2 + j+1] = d_k2[(i+4)*SIZE2 + j+1] + d_k2[(i+5)*SIZE2 + j+1] + d_k2[(i+6)*SIZE2 + j+1];
        d_k2[(i+6)*SIZE2 + j+1] = d_k2[(i+5)*SIZE2 + j+1] + d_k2[(i+6)*SIZE2 + j+1] + d_k2[(i+7)*SIZE2 + j+1];
        d_k2[(i+7)*SIZE2 + j+1] = d_k2[(i+6)*SIZE2 + j+1] + d_k2[(i+7)*SIZE2 + j+1] + d_k2[(i+8)*SIZE2 + j+1];
      }
      for(int i1=i; i1<(SIZE2-1); i1++) {
        d_k2[i1*SIZE2 + j+1] = d_k2[(i1-1)*SIZE2 + j+1] + d_k2[i1*SIZE2 + j+1] + d_k2[(i1+1)*SIZE2 + j+1];
      }
    }
  }
}

__host__ void serial(double* h_ser) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser[i*SIZE1 + j+1] =
            (h_ser[(i-1)*SIZE1 + j+1] + h_ser[i*SIZE1 + j+1] + h_ser[(i+1)*SIZE1 + j+1]);
      }
    }
  }
}

__host__ void check_result(double* w_ref, double* w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      this_diff = w_ref[i*size + j] - w_opt[i*size + j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() { // Seconds
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
  double* h_ser = new double[SIZE1*SIZE1];
  double* h_k1 = new double[SIZE1*SIZE1];   //needs to be contiguous for cudaMemcpy to work

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser[i*SIZE1 + j] = 1;
      h_k1[i*SIZE1 + j] = 1;
    }
  }

  double* h_k2 = new double[SIZE2*SIZE2];

  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {
      h_k2[i*SIZE2 + j] = 1;
    }
  }

  double clkbegin = rtclock();
  serial(h_ser);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
       << " GFLOPS; Time = " << time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float k1_time, k2_time; // milliseconds

  double* d_k1;
  // TODO: Fill in
  cudaMalloc(&d_k1, sizeof(double)*SIZE1*SIZE1);
  // full parallelization
  dim3 threads_in_block1(32);
  dim3 blocks_in_grid1(SIZE1/threads_in_block1.x);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_k1, h_k1, sizeof(double)*SIZE1*SIZE1, cudaMemcpyHostToDevice);
  kernel1<<<blocks_in_grid1, threads_in_block1>>>(d_k1);
  cudaMemcpy(h_k1, d_k1, sizeof(double)*SIZE1*SIZE1, cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k1_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_ser, h_k1, SIZE1);
  cout << "Kernel 1 on GPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  double* d_k2;
  // TODO: Fill in
  cudaMalloc(&d_k2, sizeof(double)*SIZE2*SIZE2);
  dim3 threads_in_block2(32);
  dim3 blocks_in_grid2((SIZE2+threads_in_block2.x-1)/threads_in_block2.x);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_k2, h_k2, sizeof(double)*SIZE2*SIZE2, cudaMemcpyHostToDevice);
  kernel2<<<blocks_in_grid2, threads_in_block2>>>(d_k2);
  cudaMemcpy(h_k2, d_k2, sizeof(double)*SIZE2*SIZE2, cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k2_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cout << "Kernel 2 on GPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / (k2_time * 1.0e-3))
       << " GFLOPS; Time = " << k2_time << " msec" << endl;
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << "CUDA Error: " << cudaGetErrorString(status) << endl;
  }

  cudaFree(d_k1);
  cudaFree(d_k2);

  delete[] h_ser;
  delete[] h_k1;
  delete[] h_k2;

  return EXIT_SUCCESS;
}
