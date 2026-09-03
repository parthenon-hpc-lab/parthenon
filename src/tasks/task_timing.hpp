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
#ifndef TASKS_TASK_TIMING_HPP_
#define TASKS_TASK_TIMING_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <ios>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <basic_types.hpp>
#include <parthenon_mpi.hpp>

#include "utils/error_checking.hpp"

namespace parthenon {

class Task;
class TimingAccumulator : public std::enable_shared_from_this<TimingAccumulator> {
 public:
  using time_t = std::chrono::time_point<std::chrono::steady_clock>;
  using timing_chunk_t = std::tuple<time_t, time_t, TaskStatus>;

 private:
  bool collecting{false};
  std::vector<timing_chunk_t> timings;
  Real total_time;
  int ntasks{0};

  class private_t {};

 public:
  explicit TimingAccumulator(private_t) {}

  static std::shared_ptr<TimingAccumulator> create() {
    return std::make_shared<TimingAccumulator>(private_t());
  }

  void AddTiming(const timing_chunk_t &timing);
  void StopCollectingTasks() { collecting = false; }
  void StartCollectingTasks() { collecting = true; }

  void CollectTask(Task *task);
  void CollectTaskIfCollecting(Task *task);

  double GetDurationInSeconds(time_t start, time_t end) const {
    return 1.e-9 *
           static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  }

  Real GetTotalTime() const;

  int GetTotalTasks() const { return ntasks; }

  const std::vector<timing_chunk_t> &GetTimings() const { return timings; }
};

struct TimingAccumulatorGuard {
  explicit TimingAccumulatorGuard(std::shared_ptr<TimingAccumulator> timing_accumulator)
      : tidc(timing_accumulator) {
    tidc->StartCollectingTasks();
  }
  ~TimingAccumulatorGuard() { tidc->StopCollectingTasks(); }
  std::shared_ptr<TimingAccumulator> tidc;
};

class TaskList;
class TimingAccumulatorDictionary {
  std::map<std::string, std::shared_ptr<TimingAccumulator>> dict_;

 public:
  std::shared_ptr<TimingAccumulator> GetOrAddAndRegister(const std::string &label,
                                                         TaskList &tl);

  std::shared_ptr<TimingAccumulator> Get(const std::string &label) {
    PARTHENON_REQUIRE(dict_.count(label) > 0, "Asking for non-existent timing region.");
    return dict_[label];
  }

  void clear() { dict_.clear(); }
  auto begin() { return dict_.begin(); }
  auto end() { return dict_.end(); }
  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

  void WriteToJSON(const std::string &file_name);

  friend std::ostream &operator<<(std::ostream &os,
                                  const TimingAccumulatorDictionary &tad) {
    os << std::fixed << std::setprecision(6);

    for (const auto &[name, acc] : tad.dict_) {
      os << name << ": ";
      if (acc) {
        os << acc->GetTotalTime() << " (s)";
      } else {
        os << "(null)";
      }
      os << '\n';
    }
    return os;
  }
};

} // namespace parthenon

#endif // TASKS_TASK_TIMING_HPP_
