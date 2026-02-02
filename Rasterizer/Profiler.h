#pragma once
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <format>

class Profiler {
public:
    Profiler(int targetTotalLoops = 12000, int warmUpFrames = 2000) {
        _targetTotalLoops = targetTotalLoops;
		_warmUpFrames = warmUpFrames;
        _frameTimes.reserve(targetTotalLoops + warmUpFrames);
    }

    void startFrame() {
        _start = std::chrono::high_resolution_clock::now();
        _currentLoopTimes++;
    }

    void endFrame() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - _start;
        _frameTimes.push_back(elapsed.count());
    }

    void printReport(const std::string& configName) {
        if (_frameTimes.size() <= _warmUpFrames) {
            std::cout << "Error! Data is not enough to output" << std::endl;
            return;
        }

		// select valid data range after warm-up
        auto validStart = _frameTimes.begin() + _warmUpFrames;
        auto validEnd = _frameTimes.end();
        auto validCount = validEnd - validStart;

        // mean + variance
        double mean = 0.0;
        double M2 = 0.0;
        size_t n = 0;

        for (auto it = validStart; it != validEnd; ++it) {
            n++;
            const double x = *it;
            const double delta = x - mean;
            mean += delta / static_cast<double>(n);
            const double delta2 = x - mean;
            M2 += delta * delta2;
        }

        auto variance = (n > 0) ? (M2 / static_cast<double>(n)) : 0.0;
        auto stdDev = std::sqrt(variance);


        // Total time (without warm-up)
        auto sumMs = std::accumulate(validStart, validEnd, 0.0);
        auto totalSec = sumMs / 1000.0;

        // P99 frame time (ms) via nth_element (no full sort)
        std::vector<double> tmp(validStart, validEnd);
        size_t p99Index = static_cast<size_t>(static_cast<double>(validCount) * 0.99);
        if (p99Index >= validCount) p99Index = validCount - 1;
        std::nth_element(tmp.begin(), tmp.begin() + p99Index, tmp.end());
        const double p99Ms = tmp[p99Index];

        double worstSum = 0.0;
        size_t worstCnt = 0;
        for (auto it = validStart; it != validEnd; ++it) {
            if (*it >= p99Ms) {
                worstSum += *it;
                worstCnt++;
            }
        }
        const double worstAvgMs = (worstCnt > 0) ? (worstSum / static_cast<double>(worstCnt)) : p99Ms;
        const double low1pctFps = (worstAvgMs > 0.0) ? (1000.0 / worstAvgMs) : 0.0;

        constexpr int KEY_WIDTH = 30;

        std::cout << "\n==========================================\n";
        std::cout << std::format("Report: {}\n", configName);

        //std::cout << std::format("{:<{}}: {}\n", "Loop Times", KEY_WIDTH, _frameTimes.size());
        //std::cout << std::format("{:<{}}: {}\n", "Dropped Warm-up Frames", KEY_WIDTH, _warmUpFrames);
        std::cout << std::format("{:<{}}: {}\n", "Valid Frame Count", KEY_WIDTH, validCount);
        std::cout << std::format("{:<{}}: {:.3f} s\n", "Valid Total Time", KEY_WIDTH, totalSec);

        std::cout << "------------------------------------------\n";

        // Core metrics (per your request: remove min/max, focus on variance)
        std::cout << std::format("{:<{}}: {:.4f} ms\n", "Avg Frame Time", KEY_WIDTH, mean);
        std::cout << std::format("{:<{}}: {:.6f} (ms^2)\n", "Variance", KEY_WIDTH, variance);
        std::cout << std::format("{:<{}}: {:.4f} ms\n", "Std Dev", KEY_WIDTH, stdDev);

        // Tail metrics
        std::cout << std::format("{:<{}}: {:.4f} ms\n", "P99 Frame Time", KEY_WIDTH, p99Ms);
        std::cout << std::format("{:<{}}: {:.4f}\n", "Average FPS", KEY_WIDTH, mean > 0 ? 1000.0 / mean : 0.0);
        std::cout << std::format("{:<{}}: {:.4f}\n", "Low 1% Avg FPS", KEY_WIDTH, low1pctFps);

        std::cout << "==========================================\n\n";

        clear();
    }

    void clear() {
        _frameTimes.clear();
        _currentLoopTimes = 0;
    }

    bool needToLoop() {
		return _currentLoopTimes < _targetTotalLoops;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;

    // store frame times in milliseconds
	std::vector<double> _frameTimes;

    int _currentLoopTimes = 0;

	int _targetTotalLoops = 20000;
	int _warmUpFrames = 3000;
};