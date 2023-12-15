//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

#ifndef TASKS_THREAD_POOL_HPP_
#define TASKS_THREAD_POOL_HPP_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace parthenon {

template <typename T>
class ThreadQueue {
 public:
  explicit ThreadQueue(const int num_workers) : nworkers(num_workers), nwaiting(0) {}
  void push(T q) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(q);
    cv.notify_one();
  }
  bool pop(T &q) {
    std::unique_lock<std::mutex> lock(mutex);
    if (queue.empty()) {
      nwaiting++;
      if (nwaiting == nworkers && started) {
        complete = true;
        complete_cv.notify_all();
      }
      cv.wait(lock, [this]() { return exit || !queue.empty(); });
      nwaiting--;
    }
    started = true;
    if (exit && complete && queue.empty()) return true;
    if (!queue.empty()) {
      q = queue.front();
      queue.pop();
    }
    return false;
  }
  void signal_kill() {
    std::lock_guard<std::mutex> lock(mutex);
    std::queue<T>().swap(queue);
    complete = true;
    exit = true;
    cv.notify_all();
  }
  void signal_exit_when_finished() {
    std::lock_guard<std::mutex> lock(mutex);
    exit = true;
    complete = true;
    cv.notify_all();
  }
  void wait_for_complete() {
    std::unique_lock<std::mutex> lock(mutex);
    if (complete) {
      complete = false;
      return;
    }
    complete_cv.wait(lock, [this]() { return complete; });
    complete = false;
  }

 private:
  const int nworkers;
  int nwaiting;
  std::queue<T> queue;
  std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable complete_cv;
  bool complete = false;
  bool exit = false;
  bool started = false;
};

class ThreadPool {
 public:
  explicit ThreadPool(const int numthreads = std::thread::hardware_concurrency())
      : nthreads(numthreads), queue(nthreads) {
    for (int i = 0; i < nthreads; i++) {
      auto worker = [&]() {
        while (true) {
          std::function<void()> f;
          auto stop = queue.pop(f);
          if (stop) break;
          if (f) f();
        }
      };
      threads.emplace_back(worker);
    }
  }
  ~ThreadPool() {
    queue.signal_exit_when_finished();
    for (auto &t : threads) {
      t.join();
    }
  }

  void wait() { queue.wait_for_complete(); }

  void kill() { queue.signal_kill(); }

  template <typename F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueue(F &&f, Args &&...args) {
    using return_t = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(std::forward<Args>(args)...); });
    std::future<return_t> result = task->get_future();
    queue.push([task]() { (*task)(); });
    return result;
  }

 private:
  const int nthreads;
  std::vector<std::thread> threads;
  ThreadQueue<std::function<void()>> queue;
};

} // namespace parthenon

#endif // TASKS_THREAD_POOL_HPP_
