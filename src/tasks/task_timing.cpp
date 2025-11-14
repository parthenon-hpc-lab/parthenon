//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "task_timing.hpp"
#include "tasks.hpp"

namespace parthenon {

void TimingAccumulator::CollectTask(Task *task) {
  ntasks++;
  task->time_task = true;
  task->timing_accumulators.push_back(shared_from_this());
}

void TimingAccumulator::CollectTaskIfCollecting(Task *task) {
  if (collecting) CollectTask(task);
}

Real TimingAccumulator::GetTotalTime() const {
  Real total_time{0.0};
  for (auto &[start, end, status] : timings)
    total_time += GetDurationInSeconds(start, end);
  return total_time;
}

std::shared_ptr<TimingAccumulator>
TimingAccumulatorDictionary::GetOrAddAndRegister(const std::string &label, TaskList &tl) {
  if (dict_.count(label) == 0) dict_[label] = TimingAccumulator::create();
  tl.RegisterTimingAccumulator(dict_[label]);
  return dict_[label];
}

void TimingAccumulatorDictionary::WriteToJSON(const std::string &filename) {
  std::map<std::string, std::vector<std::pair<double, double>>> timings;

  // First, find the minimum time to set zero
  TimingAccumulator::time_t min_time = std::chrono::steady_clock::now();
  for (auto &[name, taccum] : dict_) {
    for (const auto &timing : taccum->GetTimings()) {
      min_time = std::min(min_time, std::get<0>(timing));
    }
  }

  // Now, go through and build the map that can be interpreted by python
  for (auto &[name, taccum] : dict_) {
    timings[name] = std::vector<std::pair<double, double>>();
    for (const auto &timing : taccum->GetTimings()) {
      const double start = taccum->GetDurationInSeconds(min_time, std::get<0>(timing));
      const double end = taccum->GetDurationInSeconds(min_time, std::get<1>(timing));
      timings[name].push_back(std::make_pair(start, end));
    }
  }

  std::ofstream file(filename);
  file << "{";

  bool firstKey = true;
  for (const auto &[key, value] : timings) {
    if (!firstKey) {
      file << ",";
    }
    firstKey = false;
    std::cout << "Writing key " << key << std::endl;
    file << "\"" << key << "\":[";

    bool firstPair = true;
    for (const auto &pair : value) {
      if (!firstPair) {
        file << ",";
      }
      firstPair = false;

      // Write pair as JSON array [first, second]
      // Use high precision to preserve double values
      file << "[" << std::fixed << std::setprecision(15) << pair.first << ","
           << pair.second << "]";
    }

    file << "]";
  }

  file << "}";
  file.close();
}

} // namespace parthenon
