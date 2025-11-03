#include "utils/ThreadPool.h"
#include "utils/ConfigParser.h"
#include <iostream>
#include <stdexcept>
#include <thread>

ThreadPool &getThreadPool() {
  static ThreadPool pool([]() {
    return ConfigParser::getInstance().getValue<int>("num_threads");
  }());
  return pool;
}

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
  if (num_threads == 0) {
    num_threads = 1;
    std::cerr << "Warning: ThreadPool created with 0 threads, defaulting to 1."
              << std::endl;
  }
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(
              lock, [this] { return this->stop_ || !this->tasks_.empty(); });
          if (this->stop_ && this->tasks_.empty())
            return;
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }

        task();

        {
          std::unique_lock<std::mutex> lock(this->completion_mutex_);
          this->completed_tasks_++;
          this->completion_condition_.notify_one();
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  condition_.notify_all();

  for (std::thread &worker : workers_) {
    worker.join();
  }
}

void ThreadPool::run_batch(std::vector<std::function<void()>> tasks) {
  size_t batch_size = tasks.size();
  if (batch_size == 0)
    return;

  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_)
      throw std::runtime_error("run_batch called on stopped ThreadPool");

    for (auto &task : tasks) {
      tasks_.push(std::move(task));
    }
    expected_tasks_ += batch_size;
  }

  condition_.notify_all();

  {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_condition_.wait(lock, [this, expected = expected_tasks_.load()] {
      return completed_tasks_.load() >= expected;
    });
  }
}
