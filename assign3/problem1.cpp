// CS610 Assignment 3
// Author: Ayush Kumar
// Roll No: 170195

// Compile: g++ -O2 -o problem1 problem1.cpp
// Compile: g++ -O2 -march=native -o problem1 problem1.cpp (for compiling code with intrinsics)
// Execute: ./problem1

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

using namespace std;

const int N = 1 << 13;
const int Niter = 10;
const double THRESHOLD = 0.000001;

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

void reference(double** A, double* x, double* y_ref, double* z_ref) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(double* w_ref, double* w_opt) {
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

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THIS CODE
// You can create multiple versions of the optimized() function to test your changes

// Loop Interchange
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j;
//   for (j = 0; j < N; j++) {
//     for (i = 0; i < N; i++) {
//       y_opt[j] = y_opt[j] + A[i][j] * x[i];
//       z_opt[j] = z_opt[j] + A[j][i] * x[i];
//     }
//   }
// }

// Blocking
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j, it, jt;
//   const int block_size = 2;
//   for(it = 0; it < N; it += block_size) {
//     for (jt = 0; jt < N; jt += block_size) {
//       for (i = it; i < it+block_size; i++) {
//         for (j = jt; j < jt+block_size; j++) {
//           y_opt[j] = y_opt[j] + A[i][j] * x[i];
//           z_opt[j] = z_opt[j] + A[j][i] * x[i];
//         }
//       }
//     }
//   }
// }

// Blocking + Loop Interchange
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j, it, jt;
//   const int block_size = 512;
//   for (jt = 0; jt < N; jt += block_size) {
//     for(it = 0; it < N; it += block_size) {
//       for (j = jt; j < jt+block_size; j++) {
//         for (i = it; i < it+block_size; i++) {
//           y_opt[j] = y_opt[j] + A[i][j] * x[i];
//           z_opt[j] = z_opt[j] + A[j][i] * x[i];
//         }
//       }
//     }
//   }
// }

// Loop splitting + Loop Interchange (Best performance on GCC without intrinsics)
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j++) {
//       y_opt[j] = y_opt[j] + A[i][j] * x[i];
//     }
//   }
//   for (j = 0; j < N; j++) {
//     for (i = 0; i < N; i++) {
//       z_opt[j] = z_opt[j] + A[j][i] * x[i];
//     }
//   }
// }

// Blocking + Loop Splitting + Loop Interchange (Best performance on ICC with or without intrinsics)
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j, it, jt;
//   const int block_size = 2048;
//   for(it = 0; it < N; it += block_size) {
//     for (jt = 0; jt < N; jt += block_size) {
//       for (i = it; i < it+block_size; i++) {
//         for (j = jt; j < jt+block_size; j++) {
//           y_opt[j] = y_opt[j] + A[i][j] * x[i];
//           // z_opt[j] = z_opt[j] + A[j][i] * x[i];
//         }
//       }
//       for (j = jt; j < jt+block_size; j++) {
//         for (i = it; i < it+block_size; i++) {
//           // y_opt[j] = y_opt[j] + A[i][j] * x[i];
//           z_opt[j] = z_opt[j] + A[j][i] * x[i];
//         }
//       }
//     }
//   }
// }

// Loop Splitting + Loop Interchange + Intrinsics (Best performance on GCC with intrinsics)
void optimized(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j += 2) {
      __m128d r1, r2, r3, r4;
      r1 = _mm_loadu_pd(&y_opt[j]);
      r2 = _mm_loadu_pd(&A[i][j]);
      r3 = _mm_set1_pd(x[i]);
      r4 = _mm_add_pd(r1, _mm_mul_pd(r2, r3));
      _mm_store_pd(&y_opt[j], r4);
      // y_opt[j] = y_opt[j] + A[i][j] * x[i];
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i += 2) {
      __m128d r1, r2, r3, r4;
      r1 = _mm_loadu_pd(&x[i]);
      r2 = _mm_loadu_pd(&A[j][i]);
      r3 = _mm_mul_pd(r1, r2);
      r4 = _mm_hadd_pd(r3, r3);
      z_opt[j] = z_opt[j] + _mm_cvtsd_f64(r4);
      // z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}

// Blocking + Loop Splitting + Loop Interchange + Intrinsics
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j, it, jt;
//   const int block_size = 2048;
//   for(it = 0; it < N; it += block_size) {
//     for (jt = 0; jt < N; jt += block_size) {
//       for (i = it; i < it+block_size; i++) {
//         for (j = jt; j < jt+block_size; j += 2) {
//           __m128d r1, r2, r3, r4;
//           r1 = _mm_loadu_pd(&y_opt[j]);
//           r2 = _mm_loadu_pd(&A[i][j]);
//           r3 = _mm_set1_pd(x[i]);
//           r4 = _mm_add_pd(r1, _mm_mul_pd(r2, r3));
//           _mm_store_pd(&y_opt[j], r4);
//           // y_opt[j] = y_opt[j] + A[i][j] * x[i];
//         }
//       }
//       for (j = jt; j < jt+block_size; j++) {
//         for (i = it; i < it+block_size; i += 2) {
//           __m128d r1, r2, r3, r4;
//           r1 = _mm_loadu_pd(&x[i]);
//           r2 = _mm_loadu_pd(&A[j][i]);
//           r3 = _mm_mul_pd(r1, r2);
//           r4 = _mm_hadd_pd(r3, r3);
//           z_opt[j] = z_opt[j] + _mm_cvtsd_f64(r4);
//           // z_opt[j] = z_opt[j] + A[j][i] * x[i];
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

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  check_result(z_ref, z_opt);

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

  return EXIT_SUCCESS;
}
