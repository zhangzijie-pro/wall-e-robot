#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    size_t get_thread_count() const { return threads_.size(); }

private:
    using Task = std::function<void()>;

    void worker_thread();

    std::vector<std::thread> threads_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
};

template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Submit on stopped ThreadPool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return result;
}

#endif // THREAD_POOL_H