#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    using Job = std::function<void()>;

    explicit ThreadPool(int numThreads)
        : stopFlag(false), wake(0), jobCount(0) {
        workers.reserve(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back([this](std::stop_token st) { workerLoop(st); });
        }
    }

    ~ThreadPool() {
        stopFlag.store(true, std::memory_order_release);
        wake.fetch_add(1, std::memory_order_acq_rel);
        wake.notify_all();

        for (auto& t : workers) {
            t.request_stop();
        }
    }

    void submit(Job j) {
        jobCount.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push(std::move(j));
        }
        wake.fetch_add(1, std::memory_order_release);
        wake.notify_one();
    }

    void waitIdle() {
        while (jobCount.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }

private:
    void workerLoop(std::stop_token st) {
        while (!st.stop_requested()) {
            Job job;
            if (tryPop(job)) {
                job();
                onJobFinished();
                continue;
            }

            if (stopFlag.load(std::memory_order_acquire)) return;

            if (jobCount.load(std::memory_order_acquire) == 0) {

            }

            int old = wake.load(std::memory_order_acquire);
            if (hasWork()) continue;
            wake.wait(old, std::memory_order_relaxed);
        }
    }

    bool tryPop(Job& out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop();
        return true;
    }

    bool hasWork() {
        if (jobCount.load(std::memory_order_acquire) == 0) return false;
        std::lock_guard<std::mutex> lock(mtx);
        return !q.empty();
    }

    void onJobFinished() {
        auto remaining = jobCount.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (remaining == 0) {
            wake.fetch_add(1, std::memory_order_acq_rel);
            wake.notify_all();
        }
    }

private:
    std::mutex mtx;
    std::queue<Job> q;

    std::vector<std::jthread> workers;

    std::atomic<bool> stopFlag;
    std::atomic<int>  wake; 
    std::atomic<int64_t> jobCount;
};
