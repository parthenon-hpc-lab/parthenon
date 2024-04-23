//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldmy_tide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef UTILS_THREAD_POOL_HPP_
#define UTILS_THREAD_POOL_HPP_


#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"

/*
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
*/

namespace parthenon {

inline thread_local int my_tid;

/*
template <typename T>
class ConcurrentQueue {
 public:
  explicit ConcurrentQueue(const int nthreads)
      : built(false), main_waiting(false), exit(false), nqueues(nthreads+1), nsleep(0),
        mutex(nqueues), queue(nqueues) {}

  void push(T q) {
    {
      std::lock_guard<std::mutex> lock(mutex[my_tid]);
      queue[my_tid].push(q);
    }
    nqueued++;
    if (nsleep.load() > 0) {
      std::lock_guard<std::shared_mutex> lock(complete_mutex);
      sleep_cv.notify_all();
    }
  }

  void push(int i, T q) {
    {
      std::lock_guard<std::mutex> lock(mutex[i]);
      queue[i].push(q);
    }
    nqueued++;
    if (nsleep.load() > 0) {
      std::lock_guard<std::shared_mutex> lock(complete_mutex);
      sleep_cv.notify_all();
    }
  }

  void push_lazy(T q) {
    assert(my_tid == 0);
    queue[0].push(q);
    nqueued++;
  }
  void push_lazy(int i, T q) {
    queue[i].push(q);
    nqueued++;
  }

  void go() {
    sleep_cv.notify_all();
  }

  void pop(T &q) {
    if (mutex[my_tid].try_lock()) {
      if (!queue[my_tid].empty()) {
        q = queue[my_tid].front();
        queue[my_tid].pop();
        mutex[my_tid].unlock();
        nqueued--;
        return;
      }
      mutex[my_tid].unlock();
    }
    for (int i = 0; i < nqueues; i++) {
      if (mutex[i].try_lock()) {
        if (!queue[i].empty()) {
          q = queue[i].front();
          queue[i].pop();
          mutex[i].unlock();
          nqueued--;
          return;
        }
        mutex[i].unlock();
      }
    }
    return;
  }

  bool thread_sleep() {
    nsleep++;
    std::unique_lock<std::shared_mutex> lock(complete_mutex);
    if (nsleep == nqueues - 1 && main_waiting && nqueued.load() == 0) complete_cv.notify_all();
    sleep_cv.wait(lock, [this]() { return nqueued.load() > 0 || exit; });
    nsleep--;
    return exit && nqueued.load() == 0;
  }

  bool signal_if_complete() {
    if (nqueued.load() > 0) return false;
    bool should_exit = thread_sleep();
    return should_exit;
  }

  void signal_exit_when_finished() {
    std::unique_lock<std::shared_mutex> lock(complete_mutex);
    exit = true;
    sleep_cv.notify_all();
  }

  void wait_for_complete() {
    std::unique_lock<std::shared_mutex> lock(complete_mutex);
    main_waiting = true;
    complete_cv.wait(lock, [this]() { return nqueued == 0; });
  }

 private:
  bool built, main_waiting, exit;
  const int nqueues;
  std::atomic_int nsleep;
  std::vector<std::mutex> mutex;
  std::vector<std::queue<T>> queue;
  std::map<std::thread::id, int> worker_id;
  std::mutex built_mutex;
  std::condition_variable built_cv;
  std::atomic_int nqueued;
  std::shared_mutex complete_mutex;
  std::condition_variable_any complete_cv;
  std::condition_variable_any sleep_cv;
};


class ThreadPool {
 public:
  explicit ThreadPool(const int numthreads = std::thread::hardware_concurrency())
      : nthreads(numthreads), main_id(std::this_thread::get_id()), thread_idx(0), queue(nthreads), exec_space(nthreads+1) {
    
    auto worker = [&](const int i) {
      my_tid = i;
      t_exec_space = exec_space[i];
      while (true) {
        auto stop = queue.signal_if_complete();
        if(stop) break;
        std::function<void()> f;
        queue.pop(f);
        if (f) f();
      }
    };
    my_tid = 0;
    t_exec_space = exec_space[0];
    threads.reserve(nthreads);
    for (int i = 0; i < nthreads; i++) {
      threads.emplace_back(worker, i+1);
    }
    //queue.BuildMap(threads);
  }
  ~ThreadPool() {
    queue.signal_exit_when_finished();
    for (auto &t : threads) {
      t.join();
    }
  }
  void wait() {
    queue.wait_for_complete();
  }
  template <typename F, class... Args>
  std::future<typename std::invoke_result<F, Args...>::type> enqueue(F &&f, Args... args) {
    using return_t = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(args...); });
    std::future<return_t> result = task->get_future();
    if (std::this_thread::get_id() == main_id) {
        queue.push(thread_idx + 1, [task]() { (*task)(); });
        thread_idx = (thread_idx + 1) % nthreads;
    } else {
        queue.push([task]() { (*task)(); });
    }
    return result;
  }
  template <typename F, class... Args>
  std::future<typename std::invoke_result<F, Args...>::type> enqueue(const int tid, F &&f, Args... args) {
    using return_t = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(args...); });
    std::future<return_t> result = task->get_future();
    queue.push(tid, [task]() { (*task)(); });
    return result;
  }
  template <typename F, class... Args>
  std::future<typename std::invoke_result<F, Args...>::type> enqueue_lazy(F &&f, Args... args) {
    using return_t = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(args...); });
    std::future<return_t> result = task->get_future();
    if (std::this_thread::get_id() == main_id) {
      queue.push_lazy(thread_idx + 1, [task]() { (*task)(); });
      thread_idx = (thread_idx + 1) % nthreads;
    } else {
      printf("THIS IS WRONG!\n");
    }
    return result;
  }
  template <typename F, class... Args>
  std::future<typename std::invoke_result<F, Args...>::type> enqueue_lazy(const int tid, F &&f, Args... args) {
    using return_t = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(args...); });
    std::future<return_t> result = task->get_future();
    queue.push_lazy(tid, [task]() { (*task)(); });
    return result;
  }

  int size() const { return nthreads; }
  void run() { queue.go(); }
 private:
  const int nthreads;
  const std::thread::id main_id;
  int thread_idx;
  std::vector<std::thread> threads;
  ConcurrentQueue<std::function<void()>> queue;
  ExecSpace exec_space;
};

template <typename F>
void ThreadPoolLoop(ThreadPool &pool, const int is, const int ie, F &&f) {
  const int nthreads = pool.size();
  const double niter = (ie - is + 1) / (1.0*nthreads) + 1.e-10;
  auto looper = [func = std::forward<F>(f)](int t_is, int t_ie) {
    for (int i = t_is; i <= t_ie; i++) func(i);
  };
  for (int it = 0; it < nthreads; it++) {
    int start =  is + static_cast<int>(it*niter);
    int end = is + static_cast<int>((it+1)*niter) - 1;
    pool.enqueue_lazy(it+1, looper, start, end);
  }
  pool.run();
}
template <class... Args>
void ThreadPoolLoopBlocking(ThreadPool &pool, Args &&... args) {
  ThreadPoolLoop(pool, std::forward<Args>(args)...);
  pool.wait();
}

template <typename F, typename Reducer, typename reduce_t>
reduce_t ThreadPoolReduce(ThreadPool &pool, const int is, const int ie, F &&f, Reducer &&r, reduce_t init_val) {
  const int nthreads = pool.size();
  std::vector<std::future<reduce_t>> futures(nthreads);
  const double niter = (ie - is + 1) / (1.0*nthreads) + 1.e-10;
  auto looper = [=, func = std::forward<F>(f), red = std::forward<Reducer>(r)](int t_is, int t_ie) {
    reduce_t rval = 0;
    for (int i = t_is; i <= t_ie; i++) {
      reduce_t val = func(i);
      rval = red(rval, val);
    }
    return rval;
  };
  for (int it = 0; it < nthreads; it++) {
    int start =  is + static_cast<int>(it*niter);
    int end = is + static_cast<int>((it+1)*niter) - 1;
    futures[it] = pool.enqueue_lazy(it + 1, looper, start, end);
  }
  pool.run();
  pool.wait();
  reduce_t reduced_val = init_val;
  for (int it = 0; it < nthreads; it++) {
    if (!futures[it].valid()) {
      printf("Got an invalid future somehow!\n");
    }
    reduced_val = r(reduced_val, futures[it].get());
  }
  return reduced_val;
}

inline
void ThreadLoopBounds(ThreadPool &pool, const int is, const int ie, std::vector<int> &start, std::vector<int> &stop) {
  const int nthreads = pool.size();
  start.resize(nthreads);
  stop.resize(nthreads);
  const double niter = (ie - is + 1) / (1.0 * nthreads) + 1.e-10;
  for (int it = 0; it < nthreads; it++) {
    start[it] = is + static_cast<int>(it*niter);
    stop[it] = is + static_cast<int>((it+1)*niter) - 1;
  }
}
*/





template <typename T>
class ThreadQueue {
 public:
  explicit ThreadQueue(const int num_workers)
    : nworkers(num_workers), nwaiting(0), nqueued(0), nqueued0(0), queue(nworkers+1),
      mutex(nworkers+1), cv_go(nworkers+1) {}
  void push(T q) {
    //std::lock_guard<std::mutex> lock(mutex[my_tid]);
    mutex[my_tid].lock();
    queue[my_tid].push(q);
    mutex[my_tid].unlock();
    nqueued++;
    std::lock_guard<std::mutex> lock(mutex[0]);
    cv.notify_one();
    //if (my_tid == 0) {
      //nqueued0++;
      //cv.notify_one();
    //}
  }
  //void push(int i, T q) {
    //std::lock_guard<std::mutex> lock(mutex[i]);
    //queue[i].push(q);
    //nqueued++;
    //cv_go[i].notify_one();
  //}
  void pop_tag(int tag, T &q) {
    std::lock_guard<std::mutex> lock(mutex[tag]);
    if (queue[tag].empty()) return;
    q = queue[tag].front();
    queue[tag].pop();
    nqueued--;
    return;
  }
  bool pop(T &q) {
    pop_tag(0, q);
    if (q) return false;
    pop_tag(my_tid, q);
    if (q) return false;

    for (int i = 0; i <= nworkers; i++) {
      if (mutex[i].try_lock()) {
        if (!queue[i].empty()) {
          q = queue[i].front();
          queue[i].pop();
          mutex[i].unlock();
          nqueued--;
          return false;
        }
        mutex[i].unlock();
      }
    }

    //printf("thread %d going into waiting...\n", my_tid);

    std::unique_lock<std::mutex> lock(mutex[0]);
    nwaiting++;
    if (waiting && nwaiting.load() == nworkers && nqueued.load() == 0) {
      complete = true;
      complete_cv.notify_all();
    }
    cv.wait(lock, [this]() { return exit || nqueued.load() > 0; });
    nwaiting--;
    return exit;
  }
/*
    //std::unique_lock<std::mutex> lock(mutex[my_tid]);
    if (nqueued0.load() > 0) {
      std::lock_guard<std::mutex> lock(mutex[0]);
      if (!queue[0].empty()) {
        q = queue[0].front();
        queue[0].pop();
        nqueued--;
        nqueued0--;
        return false;
      }
    }
    mutex[my_tid].lock();
    if (queue[my_tid].empty()) {
      //mutex[my_tid].unlock();
      //std::unique_lock<std::mutex> lock(mutex[0]);
      //if (queue[0].empty()) {
        nwaiting++;
        if (waiting && nwaiting.load() == nworkers) {
          complete = true;
          complete_cv.notify_all();
        }
        cv_go[my_tid].wait(lock, [this]() { return exit || !queue[my_tid].empty(); });
        nwaiting--;
        if (exit) return true;
    }
    q = queue[my_tid].front();
    queue[my_tid].pop();
    mutex[my_tid].unlock();
    nqueued--;
    return false;
  }
  */
  //void signal_kill() {
    //std::lock_guard<std::mutex> lock(mutex[0]);
    //std::queue<T>().swap(queue);
    //complete = true;
    //exit = true;
    //cv.notify_all();
  //}
  void signal_exit_when_finished() {
    std::lock_guard<std::mutex> lock(mutex[0]);
    exit = true;
    complete = true;
    cv.notify_all();
  }
  void wait_for_complete() {
    std::unique_lock<std::mutex> lock(mutex[0]);
    waiting = true;
    if (nqueued.load() == 0 && nwaiting.load() == nworkers) {
      complete = false;
      waiting = false;
      return;
    }
    complete_cv.wait(lock, [this]() { return complete; });
    //printf("got complete!\n");
    complete = false;
    waiting = false;
  }
  //size_t size() {
    //std::lock_guard<std::mutex> lock(mutex[0]);
    //return queue.size();
  //}

 private:
  const int nworkers;
  std::atomic_int nwaiting;
  std::atomic_int nqueued, nqueued0;
  std::vector<std::queue<T>> queue;
  std::vector<std::mutex> mutex;
  std::vector<std::mutex> cv_go;
  std::condition_variable cv;
  std::condition_variable complete_cv;
  bool complete = false;
  bool exit = false;
  bool waiting = false;
};

class ThreadPool {
 public:
  explicit ThreadPool(const int numthreads = std::thread::hardware_concurrency())
      : nthreads(numthreads), queue(nthreads), exec_space(nthreads+1) {
    my_tid = 0;
    t_exec_space = exec_space[0];
    for (int i = 0; i < nthreads; i++) {
      auto worker = [&](const int i) {
        my_tid = i;
        t_exec_space = exec_space[i];
        while (true) {
          std::function<void()> f;
          auto stop = queue.pop(f);
          if (stop) break;
          if (f) f();
        }
      };
      threads.emplace_back(worker, i+1);
    }
  }
  ~ThreadPool() {
    queue.signal_exit_when_finished();
    for (auto &t : threads) {
      t.join();
    }
  }

  void wait() { //queue.notify_all(); 
  queue.wait_for_complete(); }

  //void kill() { queue.signal_kill(); }

  template <typename F, class... Args>
  std::future<typename std::invoke_result<F, Args...>::type> enqueue(F &&f, Args... args) {
    using return_t = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        [=, func = std::forward<F>(f)] { return func(args...); });
    std::future<return_t> result = task->get_future();
    queue.push([task]() { (*task)(); });
    return result;
  }

  int size() const { return nthreads; }

  //size_t num_queued() { return queue.size(); }

 private:
  const int nthreads;
  std::vector<std::thread> threads;
  ThreadQueue<std::function<void()>> queue;
  ExecSpace exec_space;
};

template <typename F>
void ThreadPoolLoop(ThreadPool &pool, const int is, const int ie, F &&f) {
  const int nthreads = pool.size();
  const double niter = (ie - is + 1) / (1.0*nthreads) + 1.e-10;
  auto looper = [func = std::forward<F>(f)](int t_is, int t_ie) {
    for (int i = t_is; i <= t_ie; i++) func(i);
  };
  for (int it = 0; it < nthreads; it++) {
    int start =  is + static_cast<int>(it*niter);
    int end = is + static_cast<int>((it+1)*niter) - 1;
    pool.enqueue(looper, start, end);
  }
}
template <class... Args>
void ThreadPoolLoopBlocking(ThreadPool &pool, Args &&... args) {
  ThreadPoolLoop(pool, std::forward<Args>(args)...);
  pool.wait();
}

template <typename F, typename Reducer, typename reduce_t>
reduce_t ThreadPoolReduce(ThreadPool &pool, const int is, const int ie, F &&f, Reducer &&r, reduce_t init_val) {
  const int nthreads = pool.size();
  std::vector<std::future<reduce_t>> futures(nthreads);
  const double niter = (ie - is + 1) / (1.0*nthreads) + 1.e-10;
  auto looper = [=, func = std::forward<F>(f), red = std::forward<Reducer>(r)](int t_is, int t_ie) {
    reduce_t rval = 0;
    for (int i = t_is; i <= t_ie; i++) {
      reduce_t val = func(i);
      rval = red(rval, val);
    }
    return rval;
  };
  for (int it = 0; it < nthreads; it++) {
    int start =  is + static_cast<int>(it*niter);
    int end = is + static_cast<int>((it+1)*niter) - 1;
    futures[it] = pool.enqueue(looper, start, end);
  }
  pool.wait();
  reduce_t reduced_val = init_val;
  for (int it = 0; it < nthreads; it++) {
    reduced_val = r(reduced_val, futures[it].get());
  }
  return reduced_val;
}

template <typename F, typename Reducer, typename reduce_t>
reduce_t ThreadPoolReduce2(ThreadPool &pool, const int is, const int ie, F &&f, Reducer &&r, reduce_t init_val) {
  std::vector<std::future<reduce_t>> futures(ie+1);
  auto flb = [=, func = std::forward<F>(f), red = std::forward<Reducer>(r)](int t_i) {
    reduce_t rval = init_val;
    return red(rval, func(t_i));
  };
  for (int it = is; it <= ie; it++) {
    futures[it] = pool.enqueue(flb, it);
  }
  pool.wait();
  reduce_t reduced_val = init_val;
  for (int it = is; it <= ie; it++) {
    reduced_val = r(reduced_val, futures[it].get());
  }
  return reduced_val;
}

inline
void ThreadLoopBounds(ThreadPool &pool, const int is, const int ie, std::vector<int> &start, std::vector<int> &stop) {
  const int nthreads = pool.size();
  start.resize(nthreads);
  stop.resize(nthreads);
  const double niter = (ie - is + 1) / (1.0 * nthreads) + 1.e-10;
  for (int it = 0; it < nthreads; it++) {
    start[it] = is + static_cast<int>(it*niter);
    stop[it] = is + static_cast<int>((it+1)*niter) - 1;
  }
}



} // namespace parthenon

#endif // TASKS_THREAD_POOL_HPP_
