//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file async_io.hpp
//  \brief Single background thread executing (I/O) tasks in submission order
// This file was made in part with generative AI.
#ifndef OUTPUTS_ASYNC_IO_HPP_
#define OUTPUTS_ASYNC_IO_HPP_

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

namespace parthenon {

// Tasks are executed one at a time (FIFO) so that background I/O of different outputs is
// serialized. Exceptions thrown by a task are stored in the returned future.
// The destructor executes all remaining tasks before joining the thread.
class AsyncIOQueue {
 public:
  AsyncIOQueue() : thread_([this] { Run_(); }) {}
  ~AsyncIOQueue() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }
  AsyncIOQueue(const AsyncIOQueue &) = delete;
  AsyncIOQueue &operator=(const AsyncIOQueue &) = delete;

  template <typename F>
  std::future<void> Submit(F &&task) {
    std::packaged_task<void()> ptask(std::forward<F>(task));
    auto future = ptask.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push_back(std::move(ptask));
    }
    cv_.notify_one();
    return future;
  }

 private:
  void Run_() {
    while (true) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
        if (tasks_.empty()) return; // only reached when stopping
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::packaged_task<void()>> tasks_;
  bool stop_ = false;
  std::thread thread_; // last member so that it is started after all others are ready
};

} // namespace parthenon

#endif // OUTPUTS_ASYNC_IO_HPP_
