#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

struct ThreadStats {
    int taskCount = 0;
    double   totalMs = 0.0;
    double   minMs = DBL_MAX;
    double   maxMs = 0.0;

    inline void record(double ms) {
        ++taskCount;
        totalMs += ms;
        minMs = std::min(minMs, ms);
        maxMs = std::max(maxMs, ms);
    }

    inline double avgMs() const {
        return taskCount ? totalMs / taskCount : 0.0;
    }
};

struct TaskTimer {
    ThreadStats& stats;
    std::chrono::high_resolution_clock::time_point start;

    TaskTimer(ThreadStats& s)
        : stats(s), start(std::chrono::high_resolution_clock::now()) {
    }

    ~TaskTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        stats.record(ms);
    }
};

class ThreadPool {
public:
    using Job = std::function<void()>;

    explicit ThreadPool(int numThreads)
        : stopFlag(false), wake(0), jobCount(0) {
        workers.reserve(numThreads);
        stats.resize(numThreads);

        for (int i = 0; i < numThreads; ++i) {
            const int idx = i;
            workers.emplace_back(
                [this, idx](std::stop_token st) {
                    workerLoop(idx, st);
                }
            );
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

    void submitBatch(std::vector<Job>& jobs) {
        if (jobs.empty()) return;

        jobCount.fetch_add(jobs.size(), std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mtx);
            for (auto& j : jobs) {
                q.push(std::move(j));
            }
        }
        wake.fetch_add(1, std::memory_order_release);
        wake.notify_all();
    }

    void waitIdle() {
        while (jobCount.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }

    void dumpStats() const {
        std::cout << "\n[ ThreadPool Efficiency ]\n";
        double totalTime = 0;
        int totalTasks = 0;
        for (const auto& s : stats) {
            totalTime += s.totalMs;
            totalTasks += s.taskCount;
        }

        std::cout << std::format("{:<20}: {} \n", "Thread Count", workers.size());

        double avgTaskTime = totalTasks ? totalTime / totalTasks : 0;
        std::cout << std::format("{:<20}: {:.3f} ms\n",
            "Avg Task Duration", avgTaskTime);

        double maxThreadTime = 0;
        for (const auto& s : stats) maxThreadTime = std::max(maxThreadTime, s.totalMs);
        double avgThreadTime = totalTime / stats.size();
        double imbalance = maxThreadTime / avgThreadTime;

        std::cout << std::format("{:<20}: {:.2f}x (Ideal: 1.0)\n", "Imbalance Factor", imbalance);

        if (avgTaskTime < 0.05)
            std::cout << "CRITICAL OVERHEAD (Batch your tasks!)\n";
        else if (imbalance > 1.5)
            std::cout << "SEVERE IMBALANCE (Consider smaller tiles or better binning)\n";
        else if (imbalance < 1.15 && avgTaskTime > 0.1)
            std::cout << "OPTIMAL\n";
        else
            std::cout << "ACCEPTABLE\n";

        std::cout << "==========================================\n";
    }

private:
    void workerLoop(int index, std::stop_token st) {
        ThreadStats& localStats = stats[index];

        while (!st.stop_requested()) {
            Job job;
            if (tryPop(job)) {
                TaskTimer timer(localStats);
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

    std::vector<ThreadStats> stats;

    std::atomic<bool> stopFlag;
    std::atomic<int>  wake; 
    std::atomic<int64_t> jobCount;
};
