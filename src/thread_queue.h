#pragma once

#include <deque>
#include <chrono>
#include <type_traits>
#include "thread_pool.hpp"

template <class T, class Finish>
class ThreadQueue {
  BS::thread_pool pool;
  Finish finish;
  size_t max_queue;
  size_t num_running;
  std::deque<std::pair<bool, std::future<T>>> queue;

  void CheckFinish() {
    for (size_t num_finish = 0;
         num_finish < 3 && queue.size() && (queue.front().first || IsReady(queue.front().second));
         num_finish++) {
      finish(queue.front().second.get());
      if (!queue.front().first) num_running--;
      queue.pop_front();
    }
    for (auto& item : queue) {
      if (!item.first && IsReady(item.second)) {
        num_running--;
        item.first = true;
      }
    }
  }

  template <class Rep, class Period>
  void WaitUntilAvailable(const std::chrono::duration<Rep, Period>& wait) {
    CheckFinish();
    while (num_running >= max_queue) {
      std::this_thread::sleep_for(wait);
      CheckFinish();
    }
  }

 public:
  ThreadQueue(size_t parallel, size_t max_queue, Finish&& finish) :
      pool(parallel), finish(finish), max_queue(max_queue), num_running() {
    static_assert(std::is_invocable_v<Finish, T>);
  }

  ~ThreadQueue() {
    WaitAll();
  }

  template <class Func, class Rep, class Period>
  void Push(Func&& f, const std::chrono::duration<Rep, Period>& wait) {
    static_assert(std::is_same<T, std::invoke_result_t<Func>>::value);
    WaitUntilAvailable(wait);
    queue.emplace_back(false, pool.submit(f));
    num_running++;
  }

  template <class Func> void Push(Func&& f) {
    Push(f, std::chrono::milliseconds(20));
  }

  void WaitAll() {
    while (queue.size()) {
      finish(queue.front().second.get());
      if (!queue.front().first) num_running--;
      queue.pop_front();
    }
  }
};

template <class T, class Finish>
ThreadQueue<T, Finish> MakeThreadQueue(size_t parallel, Finish&& finish, size_t max_queue = 0) {
  if (max_queue == 0) max_queue = parallel * 4 + 16;
  return ThreadQueue<T, Finish>(parallel, max_queue, std::forward<Finish>(finish));
}
