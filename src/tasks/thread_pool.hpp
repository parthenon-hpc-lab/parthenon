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

#ifndef TASKS_THREAD_POOL_HPP_
#define TASKS_THREAD_POOL_HPP_

#include <chrono>
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

class TaskList;

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
      if (waiting && nwaiting == nworkers) {
        complete = true;
        complete_cv.notify_all();
      }
      cv.wait(lock, [this]() { return exit || !queue.empty(); });
      nwaiting--;
      if (exit) return true;
    }
    q = queue.front();
    queue.pop();
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
  bool wait_for_complete(std::chrono::seconds max_time) {
    bool timeout{false};
    {
      std::unique_lock<std::mutex> lock(mutex);
      waiting = true;
      if (queue.empty() && nwaiting == nworkers) {
        complete = false;
        waiting = false;
        return timeout;
      }
      timeout = !complete_cv.wait_for(lock, max_time, [this]() { return complete; });
      complete = false;
      waiting = false;
    }
    if (timeout) signal_kill();
    return timeout;
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
  bool waiting = false;
};

template <typename T>
class ThreadVector {
 public:
  ThreadVector() {}
  explicit ThreadVector(const int size) : vec(size) {}

  void push(T q) {
    std::unique_lock<std::mutex> lock(mutex);
    vec.push_back(q);
  }
  std::vector<T> &get_vector() { return vec; }
  // Only called from host thread after wait()
  void clear() {
    std::unique_lock<std::mutex> lock(mutex);
    vec.clear();
  }

  TaskList &operator[](const int i) { return vec[i]; }

  size_t size() const { return vec.size(); }

  // Pass through the bits we use of a vector for convenience
  typename std::vector<T>::iterator begin() { return vec.begin(); }
  typename std::vector<T>::iterator end() { return vec.end(); }
  T &front() { return vec.front(); }

 private:
  std::vector<T> vec;
  std::mutex mutex;
};

class ThreadPool {
 public:
  explicit ThreadPool(const std::size_t timeout_in_seconds,
                      const int numthreads = std::thread::hardware_concurrency())
      : nthreads(numthreads), queue(nthreads), timeout_duration(timeout_in_seconds) {
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

  void kill() { queue.signal_kill(); }

  template <typename F, class... Args>
  void enqueue(F &&f, Args &&...args) {
    using return_t = typename std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(std::forward<Args>(args)...); });
    // If we're listing Prathenon "Tasks" (all current uses) keep the returns
    if constexpr (std::is_same<return_t, TaskStatus>::value) run_tasks.push(task);
    queue.push([task]() { (*task)(); });
  }

  int size() const { return nthreads; }

  // Mostly this exists to throw any exceptions,
  // but we can check returns too.
  // Would need changes for >1 failure mode
  TaskStatus wait() {
    const bool timeout = queue.wait_for_complete(timeout_duration);
    TaskStatus overall = TaskStatus::complete;
    for (auto &task : run_tasks) {
      TaskStatus task_return = task->get_future().get();
      if (task_return == TaskStatus::fail) overall = TaskStatus::fail;
    }
    run_tasks.clear();

    return timeout ? TaskStatus::fail : overall;
  }

 private:
  const int nthreads;
  std::vector<std::thread> threads;
  ThreadQueue<std::function<void()>> queue;
  ThreadVector<std::shared_ptr<std::packaged_task<TaskStatus()>>> run_tasks;
  std::chrono::seconds timeout_duration;
};

template <typename return_t = TaskStatus>
class SerialPool {
 public:
  explicit SerialPool(const std::size_t timeout_in_seconds,
                      [[maybe_unused]] const int numthreads = 1)
      : timeout_duration(timeout_in_seconds) {}

  template <typename F, class... Args>
  void enqueue(F &&f, Args &&...args) {
    auto task = [=, func = std::forward<F>(f)] {
      return func(std::forward<Args>(args)...);
    };
    queue.push(task);
  }

  int size() const { return 1; }

  TaskStatus wait() {
    TaskStatus overall = TaskStatus::complete;

    const auto start_time = std::chrono::high_resolution_clock::now();
    auto timeout_check = [start_time, timeout_duration = timeout_duration]() {
      const auto end_time = std::chrono::high_resolution_clock::now();
      const auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      return duration >= timeout_duration;
    };

    while (!queue.empty()) {
      auto f = queue.front();
      auto ret = f();
      if constexpr (std::is_same<return_t, TaskStatus>::value) {
        if (ret == TaskStatus::fail) overall = TaskStatus::fail;
      }
      queue.pop();
      if (timeout_check()) return TaskStatus::fail;
    }
    return overall;
  }

 private:
  std::queue<std::function<return_t()>> queue;
  std::chrono::seconds timeout_duration;
};

#ifdef PARTHENON_USE_SERIAL_POOL
using Pool_t = SerialPool<TaskStatus>;
#else
using Pool_t = ThreadPool;
#endif

} // namespace parthenon

#endif // TASKS_THREAD_POOL_HPP_
