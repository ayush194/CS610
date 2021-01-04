// Assignment 4 submission
// Author:
//
// Name: Ayush Kumar
// Roll No: 170195
//

// Compile: g++ -std=c++11 -fopenmp pi.cpp -o pi -ltbb

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <tbb/tbb.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

const int NUM_INTERVALS = std::numeric_limits<int>::max();

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

double omp_pi() {
  // TODO: Implement OpenMP version with minimal false sharing
  const int num_threads = 24;
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  // double partial_sums[num_threads*4] = {0.0};
  omp_set_num_threads(num_threads);
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    // int id = omp_get_thread_num();
    // partial_sums[id*4] += h * dx;
    sum += h * dx;
  }
  // for(int i = 0; i < num_threads*4; i += 4) {
  //   sum += partial_sums[i];
  // }
  double pi = 4 * sum;
  return pi;
}

class TbbPiTask {
public:
  double partial_sum;
  const static double dx;
  TbbPiTask(): partial_sum(0.0) {}
  TbbPiTask(TbbPiTask& t, tbb::split): partial_sum(0.0) {}
  ~TbbPiTask() {}
  void operator() (const tbb::blocked_range<size_t>& r) {
    double sum = partial_sum;
    for (size_t i = r.begin(); i < r.end(); i++) {
      double x = (i + 0.5) * dx;
      double h = std::sqrt(1 - x * x);
      sum += h * dx;
    }
    partial_sum = sum;
  }
  void join(const TbbPiTask& t) {
    partial_sum += t.partial_sum;
  }
};

const double TbbPiTask::dx = 1.0 / NUM_INTERVALS;

double tbb_pi() {
  // TODO: Implement TBB version with parallel algorithms
  TbbPiTask root_task;
  // int grain_size = 5;
  parallel_reduce(tbb::blocked_range<size_t>(0, NUM_INTERVALS), root_task);
  double pi = 4 * root_task.partial_sum;
  return pi;
}

int main() {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "g++ -std=c++11 pi.cpp -o pi; ./pi"
// End:
