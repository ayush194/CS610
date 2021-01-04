// CS610 Assignment 3
// Author: Ayush Kumar
// Roll No: 170195

// Compile: g++ -O2 -fopenmp -o problem3 problem3.cpp
// Execute: ./problem3

#include <cassert>
#include <iostream>
#include <omp.h>

#define N (1 << 12)
#define ITER 100

using namespace std;

void check_result(uint32_t** w_ref, uint32_t** w_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(w_ref[i][j] == w_opt[i][j]);
    }
  }
  cout << "No differences found between base and test versions\n";
}

void reference(uint32_t** A) {
  int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i < N; i++) {
      for (j = 0; j < (N - 1); j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
  }
}

// TODO: MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
// void omp_version(uint32_t** A) {
//   int i, j, k;
//   for (k = 0; k < ITER; k++) {
//     for (i = 1; i < N; i++) {
//       for (j = 0; j < (N - 1); j++) {
//         A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
//       }
//     }
//   }
// }

// kij Variant
// void omp_version(uint32_t** A) {
//   int i, j, k;
//   const int n_threads = 24;
//   omp_set_num_threads(n_threads);
//   for (k = 0; k < ITER; k++) {
//     for (i = 1; i < N; i++) {
//       #pragma omp parallel for
//       for (j = 0; j < (N - 1); j++) {
//         A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
//       }
//     }
//   }
// }

// kji Variant
// void omp_version(uint32_t** A) {
//   int i, j, k;
//   const int n_threads = 24;
//   omp_set_num_threads(n_threads);
//   for (k = 0; k < ITER; k++) {
//     #pragma omp parallel for
//     for (j = 0; j < (N - 1); j++) {
//       for (i = 1; i < N; i++) {
//         A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
//       }
//     }
//   }
// }

// jki Variant
// void omp_version(uint32_t** A) {
//   int i, j, k;
//   const int n_threads = 12;
//   omp_set_num_threads(n_threads);
//   #pragma omp parallel for
//   for (j = 0; j < (N - 1); j++) {
//     for (k = 0; k < ITER; k++) {
//       for (i = 1; i < N; i++) {
//         A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
//       }
//     }
//   }
// }

// kji Variant + Blocking
// void omp_version(uint32_t** A) {
//   int jt;
//   const int n_threads = 24;
//   const int block_size = 16;
//   omp_set_num_threads(n_threads);
//   #pragma omp parallel for
//   for (jt = 0; jt < N-1; jt += block_size) {
//     int i, j, k, it;
//     for (k = 0; k < ITER; k++) {
//       for (it = 0; it < N; it += block_size) {
//         for (j = jt; j < min(jt+block_size, N-1); j++) {
//           for (i = max(1, it); i < it+block_size; i++) {
//             A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
//           }
//         }
//       }
//     }
//   }
// }

// kji Variant + Blocking + Unrolling/Jamming j 4 times
// void omp_version(uint32_t** A) {
//   int jt;
//   const int n_threads = 24;
//   const int block_size = 16;
//   omp_set_num_threads(n_threads);
//   #pragma omp parallel for
//   for (jt = 0; jt < N-1; jt += block_size) {
//     int i, j, k, it;
//     for (k = 0; k < ITER; k++) {
//       for (it = 0; it < N; it += block_size) {
//         for (j = jt; j < min(jt+block_size, N-1); j += 4) {
//           for (i = max(1, it); i < it+block_size; i++) {
//             A[i][j+1] = A[i-1][j+1] + A[i][j+1];
//             A[i][j+2] = A[i-1][j+2] + A[i][j+2];
//             A[i][j+3] = A[i-1][j+3] + A[i][j+3];
//             A[i][j+4] = A[i-1][j+4] + A[i][j+4];
//           }
//         }
//       }
//     }
//   }
// }

// kji Variant + Blocking + Unrolling/Jamming i 8 times (Best performance on GCC)
void omp_version(uint32_t** A) {
  int jt;
  const int n_threads = 24;
  const int block_size = 16;
  omp_set_num_threads(n_threads);
  #pragma omp parallel for
  for (jt = 0; jt < N-1; jt += block_size) {
    int i, j, k, it;
    for (k = 0; k < ITER; k++) {
      for (it = 0; it < N; it += block_size) {
        for (j = jt; j < min(jt+block_size, N-1); j++) {
          if (it == 0) {
            A[1][j+1] = A[0][j+1] + A[1][j+1];
            A[2][j+1] = A[1][j+1] + A[2][j+1];
            A[3][j+1] = A[2][j+1] + A[3][j+1];
            A[4][j+1] = A[3][j+1] + A[4][j+1];
            A[5][j+1] = A[4][j+1] + A[5][j+1];
            A[6][j+1] = A[5][j+1] + A[6][j+1];
            A[7][j+1] = A[6][j+1] + A[7][j+1];
          }
          for (i = max(8, it); i < it+block_size; i += 8) {
            A[i][j+1] = A[i-1][j+1] + A[i][j+1];
            A[i+1][j+1] = A[i][j+1] + A[i+1][j+1];
            A[i+2][j+1] = A[i+1][j+1] + A[i+2][j+1];
            A[i+3][j+1] = A[i+2][j+1] + A[i+3][j+1];
            A[i+4][j+1] = A[i+3][j+1] + A[i+4][j+1];
            A[i+5][j+1] = A[i+4][j+1] + A[i+5][j+1];
            A[i+6][j+1] = A[i+5][j+1] + A[i+6][j+1];
            A[i+7][j+1] = A[i+6][j+1] + A[i+7][j+1];
          }
        }
      }
    }
  }
}

int main() {
  uint32_t** A_ref = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_ref[i] = new uint32_t[N];
  }

  uint32_t** A_omp = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_omp[i] = new uint32_t[N];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_ref[i][j] = i + j + 1;
      A_omp[i][j] = i + j + 1;
    }
  }

  double start = omp_get_wtime();
  reference(A_ref);
  double end = omp_get_wtime();
  cout << "Time for reference version: " << end - start << " seconds\n";

  start = omp_get_wtime();
  omp_version(A_omp);
  end = omp_get_wtime();
  cout << "Version1: Time with OpenMP: " << end - start << " seconds\n";
  check_result(A_ref, A_omp);

  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  // Another optimized version possibly

  return EXIT_SUCCESS;
}
