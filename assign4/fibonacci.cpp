// Assignment 4 submission
// Author:
//
// Name: Ayush Kumar
// Roll No: 170195
//

// Compile: g++ -std=c++11 -fopenmp fibonacci.cpp -o fibonacci -ltbb

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <tbb/tbb.h>

#define N 50

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

// Serial Fibonacci
long ser_fib(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}

long omp_fib_v1_helper(int n) {
  if (n <= 1) return n;
  long s1, s2;
  #pragma omp task shared(s1)
  {
    s1 = omp_fib_v1_helper(n-1);
  }
  #pragma omp task shared(s2)
  {
    s2 = omp_fib_v1_helper(n-2);
  }
  #pragma omp taskwait
  return s1 + s2;
}

long omp_fib_v1(int n) {
  // TODO: Implement OpenMP version with explicit tasks
  long sum;
  const int num_threads = 4;
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared(sum)
  {
    #pragma omp single
    {
      sum = omp_fib_v1_helper(n);
    }
  }
  return sum;
}

long omp_fib_v2_helper(int n) {
  if (n <= 30) return ser_fib(n);
  long s1, s2;
  #pragma omp task shared(s1)
  {
    s1 = omp_fib_v2_helper(n-1);
  }
  #pragma omp task shared(s2)
  {
    s2 = omp_fib_v2_helper(n-2);
  }
  #pragma omp taskwait
  return s1 + s2;
}

long omp_fib_v2(int n) {
  // TODO: Implement an optimized OpenMP version with any valid optimization
  long sum;
  const int num_threads = 4;
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    #pragma omp single
    {
      sum = omp_fib_v2_helper(n);
    }
  }
  return sum;
}

class TbbFibContTask: public tbb::task {
public:
  long* const sum;
  long s1, s2;
  TbbFibContTask(long* s_): sum(s_) {}
  tbb::task* execute() {
    *sum = s1 + s2;
    return NULL;
  }
};

class TbbFibTaskBlocking: public tbb::task {
public:
  int n;
  long* const sum;
  TbbFibTaskBlocking(int n_, long* sum_): n(n_), sum(sum_) {}
  tbb::task* execute() {
    if (n <= 30) *sum = ser_fib(n);
    else {
      long s1, s2;
      TbbFibTaskBlocking* t1 = new(tbb::task::allocate_child()) TbbFibTaskBlocking(n-1, &s1);
      TbbFibTaskBlocking* t2 = new(tbb::task::allocate_child()) TbbFibTaskBlocking(n-2, &s2);
      tbb::task::set_ref_count(3); // 2 + 1 for children + wait
      tbb::task::spawn(*t1);
      tbb::task::spawn_and_wait_for_all(*t2);
      *sum = s1 + s2;
    }
    return NULL;
  }
};

class TbbFibTaskCPS: public tbb::task {
public:
  int n;
  long* const sum;
  TbbFibTaskCPS(int n_, long* sum_): n(n_), sum(sum_) {}
  tbb::task* execute() {
    if (n <= 30) *sum = ser_fib(n);
    else {
      TbbFibContTask* t = new(tbb::task::allocate_continuation()) TbbFibContTask(sum);
      TbbFibTaskCPS* t1 = new(t->allocate_child()) TbbFibTaskCPS(n-1, &t->s1);
      TbbFibTaskCPS* t2 = new(t->allocate_child()) TbbFibTaskCPS(n-2, &t->s2);
      t->set_ref_count(2); // 2 + 0 for children + nowait
      tbb::task::spawn(*t1);
      // tbb::task::spawn(*t2);
      return t2;  // scheduler bypass
    }
    return NULL;
  }
};


long tbb_fib_blocking(int n) {
  // TODO: Implement Intel TBB version with blocking style
  long sum;
  TbbFibTaskBlocking* root_task = new(tbb::task::allocate_root()) TbbFibTaskBlocking(n, &sum);
  tbb::task::spawn_root_and_wait(*root_task);
  return sum;
}

long tbb_fib_cps(int n) {
  // TODO: Implement Intel TBB version with continuation passing style
  long sum;
  TbbFibTaskCPS* root_task = new(tbb::task::allocate_root()) TbbFibTaskCPS(n, &sum);
  tbb::task::spawn_root_and_wait(*root_task);
  return sum;
}

int main(int argc, char** argv) {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long s_fib = ser_fib(N);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v1 = omp_fib_v1(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v2 = omp_fib_v2(N);
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << endl;

  return EXIT_SUCCESS;
}
