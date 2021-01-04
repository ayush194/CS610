// CS610 Assignment 3
// Author: Ayush Kumar
// Roll No: 170195

// Compile: g++ -O2 -o problem2 problem2.cpp
// Compile: g++ -O2 -march=native -o problem2 problem2.cpp (for compiling code with intrinsics)
// Execute: ./problem2

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

using namespace std;

const int N = 1024;
const int Niter = 10;
const double THRESHOLD = 0.0000001;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** A, double** B, double** C) {
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < i + 1; k++) {
        C[i][j] += A[k][i] * B[j][k];
      }
    }
  }
}

void check_result(double** w_ref, double** w_opt) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
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

// TODO: THIS IS INITIALLY IDENTICAL TO REFERENCE. MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
// You can create multiple versions of the optimized() function to test your changes
// void optimized(double** A, double** B, double** C) {
//   int i, j, k;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j++) {
//       for (k = 0; k < i + 1; k++) {
//         C[i][j] += A[k][i] * B[j][k];
//       }
//     }
//   }
// }

// Blocking
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt;
//   const int block_size = 512;
//   for (it = 0; it < N; it += block_size) {
//     for (jt = 0; jt < N; jt += block_size) {
//       for (kt = 0; kt < it+block_size; kt += block_size) {
//         for (i = it; i < it+block_size; i++) {
//           for (j = jt; j < jt+block_size; j++) {
//             for (k = kt; k < min(i+1, kt+block_size); k++) {
//               C[i][j] += A[k][i] * B[j][k];
//             }
//           }
//         }
//       }
//     }
//   }
// }

// Loop Permutation
// void optimized(double** A, double** B, double** C) {
//   int i, j, k;
//   for (k = 0; k < N; k++) {
//     for (i = k; i < N; i++) {
//       for (j = 0; j < N; j += 4) {
//         C[i][j] += A[k][i] * B[j][k];
//         C[i][j+1] += A[k][i] * B[j+1][k];
//         C[i][j+2] += A[k][i] * B[j+2][k];
//         C[i][j+3] += A[k][i] * B[j+3][k];
//       }
//     }
//   }
// }

// jik Variant + Blocking + Unrolling/Jamming j 4 times
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
//   const int block_size = 512;
//   for (jt = 0; jt < N; jt += block_size) {
//     for (it = 0; it < N; it += block_size) {
//       for (kt = 0; kt < it+block_size; kt += block_size) {
//         for (j = jt; j < jt+block_size; j += 4) {
//           for (i = it; i < it+block_size; i++) {
//             double tmp = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
//             for (k = kt; k < min(i+1, kt+block_size); k++) {
//               // C[i][j] += A[k][i] * B[j][k];
//               double tmp4 = A[k][i];
//               tmp += tmp4 * B[j][k];
//               tmp1 += tmp4 * B[j+1][k];
//               tmp2 += tmp4 * B[j+2][k];
//               tmp3 += tmp4 * B[j+3][k];
//             }
//             C[i][j] += tmp;
//             C[i][j+1] += tmp1;
//             C[i][j+2] += tmp2;
//             C[i][j+3] += tmp3;
//           }
//         }
//       }
//     }
//   }
// }

// ijk Variant + Blocking + Unrolling/Jamming j 4 times + Intrinsics (Best performance on GCC with intrinsics)
void optimized(double** A, double** B, double** C) {
  int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
  const int block_size = 32;
  for (it = 0; it < N; it += block_size) {
    for (jt = 0; jt < N; jt += block_size) {
      for (kt = 0; kt < it+block_size; kt += block_size) {
        for (i = it; i < it+block_size; i++) {
          for (j = jt; j < jt+block_size; j += 4) {
            __m128d tmp = {0, 0}, tmp1 = {0, 0};
            for (k = kt; k+1 < min(i+1, kt+block_size); k += 2) {
              // C[i][j] += A[k][i] * B[j][k];
              // vectorized version
              __m128d r4 = {A[k][i], A[k+1][i]};
              __m128d r = _mm_loadu_pd(&B[j][k]);
              __m128d r1 = _mm_loadu_pd(&B[j+1][k]);
              __m128d r2 = _mm_loadu_pd(&B[j+2][k]);
              __m128d r3 = _mm_loadu_pd(&B[j+3][k]);
              r = _mm_mul_pd(r4, r);
              r1 = _mm_mul_pd(r4, r1);
              r2 = _mm_mul_pd(r4, r2);
              r3 = _mm_mul_pd(r4, r3);
              r = _mm_hadd_pd(r, r1);
              r1 = _mm_hadd_pd(r2, r3);
              tmp = _mm_add_pd(tmp, r);
              tmp1 = _mm_add_pd(tmp1, r1);
            }
            if (k < min(i+1, kt+block_size)) {
              double tmp4 = A[k][i];
              __m128d r = {tmp4 * B[j][k], tmp4 * B[j+1][k]};
              __m128d r1 = {tmp4 * B[j+2][k], tmp4 * B[j+3][k]};
              tmp = _mm_add_pd(tmp, r);
              tmp1 = _mm_add_pd(tmp1, r1);
            }
            __m128d r = _mm_loadu_pd(&C[i][j]);
            __m128d r1 = _mm_loadu_pd(&C[i][j+2]);
            r = _mm_add_pd(r, tmp);
            r1 = _mm_add_pd(r1, tmp1);
            _mm_store_pd(&C[i][j], r);
            _mm_store_pd(&C[i][j+2], r1);
          }
        }
      }
    }
  }
}

// ikj Variant + Blocking + Unrolling/Jamming j 4 times
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
//   const int block_size = 64;
//   for (it = 0; it < N; it += block_size) {
//     for (kt = 0; kt < it+block_size; kt += block_size) {
//       for (jt = 0; jt < N; jt += block_size) {
//         for (i = it; i < it+block_size; i++) {
//           for (k = kt; k < min(i+1, kt+block_size); k++) {
//             double tmp = A[k][i];
//             for (j = jt; j < jt+block_size; j += 4) {
//               // C[i][j] += A[k][i] * B[j][k];
//               C[i][j] += tmp * B[j][k];
//               C[i][j+1] += tmp * B[j+1][k];
//               C[i][j+2] += tmp * B[j+2][k];
//               C[i][j+3] += tmp * B[j+3][k];
//             }
//           }
//         }
//       }
//     }
//   }
// }

// ijk Variant + Blocking + Unrolling/Jamming j 4 times 
// (Block size 32 gives best performance on GCC without intrinsics)
// (Block size 128 gives best performance on ICC with or without intrinsics)
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
//   const int block_size = 128;
//   for (it = 0; it < N; it += block_size) {
//     for (jt = 0; jt < N; jt += block_size) {
//       for (kt = 0; kt < it+block_size; kt += block_size) {
//         for (i = it; i < it+block_size; i++) {
//           for (j = jt; j < jt+block_size; j += 4) {
//             double tmp = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
//             for (k = kt; k < min(i+1, kt+block_size); k++) {
//               // C[i][j] += A[k][i] * B[j][k];
//               double tmp4 = A[k][i];
//               tmp += tmp4 * B[j][k];
//               tmp1 += tmp4 * B[j+1][k];
//               tmp2 += tmp4 * B[j+2][k];
//               tmp3 += tmp4 * B[j+3][k];
//             }
//             C[i][j] += tmp;
//             C[i][j+1] += tmp1;
//             C[i][j+2] += tmp2;
//             C[i][j+3] += tmp3;
//           }
//         }
//       }
//     }
//   }
// }

// kij Variant + Blocking + Unrolling/Jamming j 4 times
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
//   const int block_size = 512;
//   for (kt = 0; kt < N; kt += block_size) {
//     for (it = kt; it < N; it += block_size) {
//       for (jt = 0; jt < N; jt += block_size) {
//         for (k = kt; k < min(it+block_size, kt+block_size); k++) {
//           for (i = max(it, k); i < it+block_size; i++) {
//             double tmp = A[k][i];
//             for (j = jt; j < jt+block_size; j += 4) {
//               // C[i][j] += A[k][i] * B[j][k];
//               C[i][j] += tmp * B[j][k];
//               C[i][j+1] += tmp * B[j+1][k];
//               C[i][j+2] += tmp * B[j+2][k];
//               C[i][j+3] += tmp * B[j+3][k];
//             }
//           }
//         }
//       }
//     }
//   }
// }

// kji Variant + Blocking + Unrolling/Jamming j 4 times
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
//   const int block_size = 512;
//   for (kt = 0; kt < N; kt += block_size) {
//     for (jt = 0; jt < N; jt += block_size) {
//       for (it = kt; it < N; it += block_size) {
//         for (k = kt; k < min(it+block_size, kt+block_size); k++) {
//           for (j = jt; j < jt+block_size; j += 4) {
//             double tmp = B[j][k], tmp1 = B[j+1][k], tmp2 = B[j+2][k], tmp3 = B[j+3][k];
//             for (i = max(it, k); i < it+block_size; i++) {
//               // C[i][j] += A[k][i] * B[j][k];
//               double tmp4 = A[k][i];
//               C[i][j] += tmp4 * tmp;
//               C[i][j+1] += tmp4 * tmp1;
//               C[i][j+2] += tmp4 * tmp2;
//               C[i][j+3] += tmp4 * tmp3;
//             }
//           }
//         }
//       }
//     }
//   }
// }

// jki Variant + Blocking + Unrolling/Jamming j 4 times
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, it, jt, kt, i_max, j_max, k_max, kt_max;
//   const int block_size = 32;
//   for (jt = 0; jt < N; jt += block_size) {
//     for (kt = 0; kt < N; kt += block_size) {
//       for (it = kt; it < N; it += block_size) {
//         for (j = jt; j < jt+block_size; j += 4) {
//           for (k = kt; k < min(it+block_size, kt+block_size); k++) {
//             double tmp = B[j][k], tmp1 = B[j+1][k], tmp2 = B[j+2][k], tmp3 = B[j+3][k];
//             for (i = max(it, k); i < it+block_size; i++) {
//               // C[i][j] += A[k][i] * B[j][k];
//               double tmp4 = A[k][i];
//               C[i][j] += tmp4 * tmp;
//               C[i][j+1] += tmp4 * tmp1;
//               C[i][j+2] += tmp4 * tmp2;
//               C[i][j+3] += tmp4 * tmp3;
//             }
//           }
//         }
//       }
//     }
//   }
// }


int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double **A, **B, **C_ref, **C_opt;
  A = new double*[N];
  B = new double*[N];
  C_ref = new double*[N];
  C_opt = new double*[N];
  for (i = 0; i < N; i++) {
    A[i] = new double[N];
    B[i] = new double[N];
    C_ref[i] = new double[N];
    C_opt[i] = new double[N];
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = i + j + 1;
      B[i][j] = (i + 1) * (j + 1);
      C_ref[i][j] = 0.0;
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    reference(A, B, C_ref);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 2.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);

  // Reset
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C_opt[i][j] = 0.0;
    }
  }

  return EXIT_SUCCESS;
}
