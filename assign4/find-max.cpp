// Assignment 4 submission
// Author:
//
// Name: Ayush Kumar
// Roll No: 170195
//

// Compile: g++ -std=c++11 find-max.cpp -o find-max -ltbb

#include <cassert>
#include <chrono>
#include <iostream>
#include <tbb/tbb.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 26)

uint32_t serial_find_max(const uint32_t* a) {
  uint32_t value_of_max = 0;
  uint32_t index_of_max = -1;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

class TbbFindMaxTask {
public:
  const uint32_t* arr;
  uint32_t partial_max_idx;

  TbbFindMaxTask(const uint32_t* a): arr(a), partial_max_idx(-1) {}
  TbbFindMaxTask(TbbFindMaxTask& t, tbb::split): arr(t.arr), partial_max_idx(-1) {}
  ~TbbFindMaxTask() {}
  void operator() (const tbb::blocked_range<size_t>& r) {
    uint32_t max_idx = partial_max_idx == -1 ? r.begin() : partial_max_idx;
    uint32_t max_val = arr[max_idx];
    for (size_t i = r.begin(); i < r.end(); i++) {
      if (arr[i] > max_val) {max_idx = i; max_val = arr[i];}
      else if (arr[i] == max_val) {max_idx = std::min(uint32_t(i), max_idx);}
    }
    partial_max_idx = max_idx;
  }
  void join(const TbbFindMaxTask& t) {
    if (t.arr[t.partial_max_idx] > arr[partial_max_idx]) partial_max_idx = t.partial_max_idx;
    else if (t.arr[t.partial_max_idx] == arr[partial_max_idx]) partial_max_idx = std::min(partial_max_idx, t.partial_max_idx);
  }
};

uint32_t tbb_find_max(const uint32_t* a) {
  // TODO: Implement a parallel max function with Intel TBB
  TbbFindMaxTask root_task(a);
  // int grain_size = 5;
  parallel_reduce(tbb::blocked_range<size_t>(0, N), root_task);
  return a[root_task.partial_max_idx];
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}
