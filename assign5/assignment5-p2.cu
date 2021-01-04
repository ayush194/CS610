// Author: Ayush Kumar
// Roll No: 170195
// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p2.cu -o assignment5-p2

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <atomic>

#define THRESHOLD (0.000001)
#define BLOCKSIZE 128
#define CPT 4096
#define FAC 8

using std::cerr;
using std::cout;
using std::endl;

__host__ void host_excl_prefix_sum(float* h_A, float* h_O, int N) {
  h_O[0] = 0;
  for (int i = 1; i < N; i++) {
    h_O[i] = h_O[i - 1] + h_A[i - 1];
  }
} 

__global__ void kernel_excl_prefix_sum_init(float* d_in, float* d_out) {
  int j1 = (blockIdx.x * BLOCKSIZE + threadIdx.x) * CPT;
  float tmp = d_out[j1];
  int unroll = 4;
  int j;
  for (j = j1+1; j+unroll-1 < j1+CPT; j += unroll) {
    tmp = d_in[j-1] + tmp;
    d_out[j] = tmp;
    tmp = d_in[j] + tmp;
    d_out[j+1] = tmp;
    tmp = d_in[j+1] + tmp;
    d_out[j+2] = tmp;
    tmp = d_in[j+2] + tmp;
    d_out[j+3] = tmp;
  }
  for(int j2 = j; j2 < j1+CPT; j2++){
    tmp = d_in[j2-1] + tmp;
    d_out[j2] = tmp;
  }

}

__global__ void kernel_excl_prefix_sum(float* d_in, float* d_out, int N, uint64_t block_coverage) {
  // TODO: Fill in
  int thread_j = (blockIdx.x * BLOCKSIZE + threadIdx.x);
  // every 2*block_coverage cells have block_coverage/8 threads working on them
  int j = (FAC*thread_j/block_coverage)*2*block_coverage + block_coverage + (thread_j % (block_coverage/FAC)) * FAC;
  int base = d_out[j - j%block_coverage-1] + d_in[j-j%block_coverage -1];

  d_out[j] += base;
  d_out[j+1] += base;
  d_out[j+2] += base;
  d_out[j+3] += base;
  d_out[j+4] += base;
  d_out[j+5] += base;
  d_out[j+6] += base;
  d_out[j+7] += base;
}

__host__ void check_result(float* w_ref, float* w_opt, int N) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
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
  const int N = (1 << 24);
  size_t size = N * sizeof(float);

  float* h_in = (float*)malloc(size);
  std::fill_n(h_in, N, 1);

  float* h_excl_sum_out = (float*)malloc(size);
  std::fill_n(h_excl_sum_out, N, 0);

  double clkbegin = rtclock();
  host_excl_prefix_sum(h_in, h_excl_sum_out, N);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial time on CPU: " << time * 1000 << " msec" << endl;

  float* h_dev_result = (float*)malloc(size);
  std::fill_n(h_dev_result, N, 0);
  float* d_in;
  float* d_out;
  cudaError_t status;
  cudaEvent_t start, end;
  float k_time; // milliseconds
  // TODO: Fill in
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  dim3 threads_in_block(BLOCKSIZE);
  dim3 blocks_in_grid(N/(BLOCKSIZE*CPT));
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  /************** CUDA **************/
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, size);
  kernel_excl_prefix_sum_init<<<blocks_in_grid, threads_in_block>>>(d_in, d_out);
  uint64_t block_coverage = CPT;
  while(block_coverage < N) {
    blocks_in_grid = (N/(BLOCKSIZE*2*FAC));
    kernel_excl_prefix_sum<<<blocks_in_grid, threads_in_block>>>(d_in, d_out, N, block_coverage);
    block_coverage *= 2;
  }
  cudaMemcpy(h_dev_result, d_out, size, cudaMemcpyDeviceToHost);
  /************** CUDA **************/
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&k_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  // for (int i = 0; i < N; i++) {
  //   cout << h_excl_sum_out[i] << ' ';
  // }
  // cout << endl;
  // for(int i = 0; i < N; i++) {
  //   cout << h_dev_result[i] << ' ';
  // }
  // cout << endl;
  check_result(h_excl_sum_out, h_dev_result, N);
  cout << "Kernel time on GPU: " << k_time << " msec" << endl;
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cerr << "CUDA Error: " << cudaGetErrorString(status) << endl;
  }

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);

  free(h_in);
  free(h_excl_sum_out);
  free(h_dev_result);

  return EXIT_SUCCESS;
}
