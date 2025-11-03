#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads);

  ~ThreadPool();

  void run_batch(std::vector<std::function<void()>> tasks);

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
  std::mutex completion_mutex_;
  std::condition_variable completion_condition_;
  std::atomic<size_t> completed_tasks_ = 0;
  std::atomic<size_t> expected_tasks_ = 0;
};

ThreadPool &getThreadPool();

#endif // THREAD_POOL_H
