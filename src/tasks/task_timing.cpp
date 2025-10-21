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

#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
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

} // namespace parthenon
