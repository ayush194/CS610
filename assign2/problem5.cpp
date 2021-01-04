/**
 * Author: Ayush Kumar
 * Roll No: 170195
 */

/**
 * g++ -o problem5 problem5.cpp -lpthread
 * ./problem5
 */

// TODO: This file is just a template, feel free to modify it to suit your needs

#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sys/time.h>

using std::cout;
using std::endl;

const uint16_t NUM_THREADS = 4; // TODO: You may want to change this
const uint16_t MAT_SIZE = 4096;

void sequential_matmul();
void* parallel_matmul(void*);
void sequential_matmul_opt();
void* parallel_matmul_opt(void*);

double rtclock();
void check_result(uint64_t*, uint64_t*);
const double THRESHOLD = 0.0000001;

uint64_t* matrix_A;
uint64_t* matrix_B;
uint64_t* sequential_C;
uint64_t* sequential_opt_C;
uint64_t* parallel_C;
uint64_t* parallel_opt_C;

uint16_t block_size;

void* parallel_matmul(void* thread_id) {
  int id = *((int*)thread_id);

  int i, j, k, n_i = MAT_SIZE / NUM_THREADS;
  int start_i = id*n_i, end_i = start_i + n_i;
  for (i = start_i; i < end_i; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      for (k = 0; k < MAT_SIZE; k++) {
        parallel_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
      }
    }
  }
  return nullptr;
}

void* parallel_matmul_opt(void* thread_id) { 
  int id = *((int*)thread_id);

  // most optimized parallel code (32x32x32 block size, unroll/jam j with step 16, num_threads = 4)
  int i, j, k, n_i = MAT_SIZE / NUM_THREADS;
  int start_i = id*n_i, end_i = start_i + n_i;
  for (int it = start_i; it < end_i; it += block_size) {
    for (int jt = 0; jt < MAT_SIZE; jt += block_size) {
      for (int kt = 0; kt < MAT_SIZE; kt += block_size) {  
        for (i = it; i < it + block_size; i++) {
          for (j = jt; j < jt + block_size; j += 16) {
            for (k = kt; k < kt + block_size; k++) {
              int a = i * MAT_SIZE, b = k * MAT_SIZE;
              parallel_opt_C[a + j] += (matrix_A[a + k] * matrix_B[b + j]);
              parallel_opt_C[a + j+1] += (matrix_A[a + k] * matrix_B[b + j+1]);
              parallel_opt_C[a + j+2] += (matrix_A[a + k] * matrix_B[b + j+2]);
              parallel_opt_C[a + j+3] += (matrix_A[a + k] * matrix_B[b + j+3]);
              parallel_opt_C[a + j+4] += (matrix_A[a + k] * matrix_B[b + j+4]);
              parallel_opt_C[a + j+5] += (matrix_A[a + k] * matrix_B[b + j+5]);
              parallel_opt_C[a + j+6] += (matrix_A[a + k] * matrix_B[b + j+6]);
              parallel_opt_C[a + j+7] += (matrix_A[a + k] * matrix_B[b + j+7]);
              parallel_opt_C[a + j+8] += (matrix_A[a + k] * matrix_B[b + j+8]);
              parallel_opt_C[a + j+9] += (matrix_A[a + k] * matrix_B[b + j+9]);
              parallel_opt_C[a + j+10] += (matrix_A[a + k] * matrix_B[b + j+10]);
              parallel_opt_C[a + j+11] += (matrix_A[a + k] * matrix_B[b + j+11]);
              parallel_opt_C[a + j+12] += (matrix_A[a + k] * matrix_B[b + j+12]);
              parallel_opt_C[a + j+13] += (matrix_A[a + k] * matrix_B[b + j+13]);
              parallel_opt_C[a + j+14] += (matrix_A[a + k] * matrix_B[b + j+14]);
              parallel_opt_C[a + j+15] += (matrix_A[a + k] * matrix_B[b + j+15]);
            }
          }
        }
      }
    }
  }

  // blocking
  // int n_i = MAT_SIZE / NUM_THREADS;
  // int start_i = id*n_i, end_i = start_i + n_i;
  // for (int it = start_i; it < end_i; it += block_size) {
  //   for (int jt = 0; jt < MAT_SIZE; jt += block_size) {
  //     for (int kt = 0; kt < MAT_SIZE; kt += block_size) {  
  //       for (i = it; i < it + block_size; i++) {
  //         for (j = jt; j < jt + block_size; j++) {
  //           for (k = kt; k < kt + block_size; k++) {
  //             parallel_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // unroll loop j with step size 4
  // int n_i = MAT_SIZE / NUM_THREADS;
  // int start_i = id*n_i, end_i = start_i + n_i;
  // for (i = start_i; i < end_i; i++) {
  //   for (j = 0; j < MAT_SIZE; j += 4) {
  //     for (k = 0; k < MAT_SIZE; k++) {
  //       parallel_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[i * MAT_SIZE + j+1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[i * MAT_SIZE + j+2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[i * MAT_SIZE + j+3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //     }
  //   }
  // }

  // blocking + unroll loop j with step size 8
  // int n_i = MAT_SIZE / NUM_THREADS;
  // int start_i = id*n_i, end_i = start_i + n_i;
  // for (int it = start_i; it < end_i; it += block_size) {
  //   for (int jt = 0; jt < MAT_SIZE; jt += block_size) {
  //     for (int kt = 0; kt < MAT_SIZE; kt += block_size) {  
  //       for (i = it; i < it + block_size; i++) {
  //         for (j = jt; j < jt + block_size; j += 8) {
  //           for (k = kt; k < kt + block_size; k++) {
  //             parallel_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //             parallel_opt_C[i * MAT_SIZE + j+1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //             parallel_opt_C[i * MAT_SIZE + j+2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //             parallel_opt_C[i * MAT_SIZE + j+3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //             parallel_opt_C[i * MAT_SIZE + j+4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //             parallel_opt_C[i * MAT_SIZE + j+5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //             parallel_opt_C[i * MAT_SIZE + j+6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //             parallel_opt_C[i * MAT_SIZE + j+7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // unroll loop (i, j) with step size (8, 8)
  // int n_i = MAT_SIZE / NUM_THREADS;
  // int start_i = id*n_i, end_i = start_i + n_i;
  // for (i = start_i; i < end_i; i += 8) {
  //   for (j = 0; j < MAT_SIZE; j += 8) {
  //     for (k = 0; k < MAT_SIZE; k++) {
  //       parallel_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       parallel_opt_C[i * MAT_SIZE + j+1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+1] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+1] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+1] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+1] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+1] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+1] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+1] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       parallel_opt_C[i * MAT_SIZE + j+2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+2] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+2] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+2] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+2] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+2] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+2] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+2] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       parallel_opt_C[i * MAT_SIZE + j+3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+3] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+3] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+3] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+3] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+3] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+3] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+3] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       parallel_opt_C[i * MAT_SIZE + j+4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+4] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+4] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+4] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+4] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+4] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+4] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+4] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       parallel_opt_C[i * MAT_SIZE + j+5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+5] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+5] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+5] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+5] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+5] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+5] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+5] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       parallel_opt_C[i * MAT_SIZE + j+6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+6] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+6] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+6] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+6] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+6] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+6] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+6] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       parallel_opt_C[i * MAT_SIZE + j+7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+1) * MAT_SIZE + j+7] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+2) * MAT_SIZE + j+7] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+3) * MAT_SIZE + j+7] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+4) * MAT_SIZE + j+7] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+5) * MAT_SIZE + j+7] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+6) * MAT_SIZE + j+7] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       parallel_opt_C[(i+7) * MAT_SIZE + j+7] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //     }
  //   }
  // }

  return nullptr;
}

void sequential_matmul() {
  int i, j, k;
  for (i = 0; i < MAT_SIZE; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      uint16_t temp = 0;
      for (k = 0; k < MAT_SIZE; k++)
        temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
      sequential_C[i * MAT_SIZE + j] = temp;
    }
  }
}

void sequential_matmul_opt() {
  int i, j, k;

  // Most optimized sequential code (32x32x32 block size, unroll/jam j with step 16
  for (int it = 0; it < MAT_SIZE; it += block_size) {
    for (int jt = 0; jt < MAT_SIZE; jt += block_size) {
      for (int kt = 0; kt < MAT_SIZE; kt += block_size) {
        for (int i = it; i < it + block_size; i++) {
          for (int j = jt; j < jt + block_size; j += 16) {
            for (int k = kt; k < kt + block_size; k++) {
              sequential_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
              sequential_opt_C[i * MAT_SIZE + j+1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
              sequential_opt_C[i * MAT_SIZE + j+2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
              sequential_opt_C[i * MAT_SIZE + j+3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
              sequential_opt_C[i * MAT_SIZE + j+4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
              sequential_opt_C[i * MAT_SIZE + j+5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
              sequential_opt_C[i * MAT_SIZE + j+6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
              sequential_opt_C[i * MAT_SIZE + j+7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
              sequential_opt_C[i * MAT_SIZE + j+8] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+8]);
              sequential_opt_C[i * MAT_SIZE + j+9] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+9]);
              sequential_opt_C[i * MAT_SIZE + j+10] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+10]);
              sequential_opt_C[i * MAT_SIZE + j+11] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+11]);
              sequential_opt_C[i * MAT_SIZE + j+12] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+12]);
              sequential_opt_C[i * MAT_SIZE + j+13] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+13]);
              sequential_opt_C[i * MAT_SIZE + j+14] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+14]);
              sequential_opt_C[i * MAT_SIZE + j+15] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+15]);
            }
          }
        }
      }
    }
  }

  // Use 3-D blocking to improve data locality
  // for (int it = 0; it < MAT_SIZE; it += block_size) {
  //   for (int jt = 0; jt < MAT_SIZE; jt += block_size) {
  //     for (int kt = 0; kt < MAT_SIZE; kt += block_size) {
  //       for (int i = it; i < it + block_size; i++) {
  //         for (int j = jt; j < jt + block_size; j++) {
  //           for (int k = kt; k < kt + block_size; k++) {
  //             sequential_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // 2-D blocking
  // for (int it = 0; it < MAT_SIZE; it += block_size) {
  //   for (int jt = 0; jt < MAT_SIZE; jt += block_size) {
  //     for (int i = it; i < it + block_size; i++) {
  //       for (int j = jt; j < jt + block_size; j++) {
  //         for (int k = 0; k < MAT_SIZE; k++) {
  //           sequential_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //         }
  //       }
  //     }
  //   }
  // }

  // unroll loop i and j
  // for (i = 0; i < MAT_SIZE; i += 8) {
  //   for (j = 0; j < MAT_SIZE; j += 8) {
  //     for (k = 0; k < MAT_SIZE; k++) {
  //       sequential_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[i * MAT_SIZE + j+1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+1] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+1] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+1] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+1] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+1] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+1] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+1] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[i * MAT_SIZE + j+2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+2] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+2] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+2] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+2] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+2] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+2] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+2] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[i * MAT_SIZE + j+3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+3] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+3] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+3] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+3] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+3] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+3] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+3] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[i * MAT_SIZE + j+4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+4] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+4] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+4] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+4] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+4] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+4] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+4] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[i * MAT_SIZE + j+5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+5] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+5] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+5] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+5] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+5] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+5] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+5] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[i * MAT_SIZE + j+6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+6] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+6] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+6] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+6] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+6] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+6] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+6] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[i * MAT_SIZE + j+7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j+7] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j+7] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j+7] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j+7] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j+7] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j+7] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j+7] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //     }
  //   }
  // }

  // unroll loop i
  // for (i = 0; i < MAT_SIZE; i += 8) {
  //   for (j = 0; j < MAT_SIZE; j++) {
  //     for (k = 0; k < MAT_SIZE; k++) {
  //       sequential_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+1) * MAT_SIZE + j] += (matrix_A[(i+1) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+2) * MAT_SIZE + j] += (matrix_A[(i+2) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+3) * MAT_SIZE + j] += (matrix_A[(i+3) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+4) * MAT_SIZE + j] += (matrix_A[(i+4) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+5) * MAT_SIZE + j] += (matrix_A[(i+5) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+6) * MAT_SIZE + j] += (matrix_A[(i+6) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[(i+7) * MAT_SIZE + j] += (matrix_A[(i+7) * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //     }
  //   }
  // }

  // unroll loop j
  // for (i = 0; i < MAT_SIZE; i++) {
  //   for (j = 0; j < MAT_SIZE; j += 16) {
  //     for (k = 0; k < MAT_SIZE; k++) {
  //       sequential_opt_C[i * MAT_SIZE + j] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
  //       sequential_opt_C[i * MAT_SIZE + j+1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]);
  //       sequential_opt_C[i * MAT_SIZE + j+2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]);
  //       sequential_opt_C[i * MAT_SIZE + j+3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]);
  //       sequential_opt_C[i * MAT_SIZE + j+4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]);
  //       sequential_opt_C[i * MAT_SIZE + j+5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]);
  //       sequential_opt_C[i * MAT_SIZE + j+6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]);
  //       sequential_opt_C[i * MAT_SIZE + j+7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]);
  //       sequential_opt_C[i * MAT_SIZE + j+8] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+8]);
  //       sequential_opt_C[i * MAT_SIZE + j+9] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+9]);
  //       sequential_opt_C[i * MAT_SIZE + j+10] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+10]);
  //       sequential_opt_C[i * MAT_SIZE + j+11] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+11]);
  //       sequential_opt_C[i * MAT_SIZE + j+12] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+12]);
  //       sequential_opt_C[i * MAT_SIZE + j+13] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+13]);
  //       sequential_opt_C[i * MAT_SIZE + j+14] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+14]);
  //       sequential_opt_C[i * MAT_SIZE + j+15] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+15]);
  //     }
  //   }
  // }

}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    printf("Error return from gettimeofday: %d\n", stat);
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void check_result(uint64_t* first_res, uint64_t* second_res) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < MAT_SIZE; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      this_diff = first_res[i * MAT_SIZE + j] - second_res[i * MAT_SIZE + j];
      if (this_diff < 0)
        this_diff = -1.0 * this_diff;
      if (this_diff > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main(int argc, char* argv[]) {
  if (argc == 2) {
    block_size = atoi(argv[1]);
  } else {
    block_size = 4;
    cout << "Using default block size = 4\n";
  }

  matrix_A = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  matrix_B = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  sequential_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  sequential_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  parallel_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  parallel_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];

  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      matrix_A[(i * MAT_SIZE) + j] = 1;
      matrix_B[(i * MAT_SIZE) + j] = 1;
      sequential_C[(i * MAT_SIZE) + j] = 0;
      sequential_opt_C[(i * MAT_SIZE) + j] = 0;
      parallel_C[(i * MAT_SIZE) + j] = 0;
      parallel_opt_C[(i * MAT_SIZE) + j] = 0;
    }
  }
  pthread_t thread_arr[NUM_THREADS];

  double clkbegin, clkend;

  clkbegin = rtclock();
  //----------------------------Sequential MM----------------------------//
  // sequential_matmul();
  //----------------------------Sequential MM----------------------------//
  //-----------------------Optimized Sequential MM-----------------------//
  sequential_matmul_opt();  
  //-----------------------Optimized Sequential MM-----------------------//
  clkend = rtclock();
  cout << "Time for Sequential version: " << (clkend - clkbegin) << "seconds.\n";

  clkbegin = rtclock();
  //---------------------------Parallelized MM---------------------------//
  // for (int i = 0; i < NUM_THREADS; i++) {
  //   int* i_copy = new int;
  //   *i_copy = i;
  //   pthread_create(&thread_arr[i], NULL, parallel_matmul, (void*)i_copy);
  // }

  // for (int i = 0; i < NUM_THREADS; i++) {
  //   pthread_join(thread_arr[i], NULL);
  // }
  //---------------------------Parallelized MM---------------------------//

  //----------------------Optimized Parallelized MM----------------------//
  for (int i = 0; i < NUM_THREADS; i++) {
    int* i_copy = new int;
    *i_copy = i;
    pthread_create(&thread_arr[i], NULL, parallel_matmul_opt, (void*)i_copy);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(thread_arr[i], NULL);
  }
  //----------------------Optimized Parallelized MM----------------------//
  clkend = rtclock();
  cout << "Time for parallel version: " << (clkend - clkbegin) << "seconds.\n";

  check_result(sequential_C, parallel_C);
  check_result(sequential_C, sequential_opt_C);
  check_result(sequential_C, parallel_opt_C);
  return 0;
}
