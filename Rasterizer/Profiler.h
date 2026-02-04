#pragma once
#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <format>
#include <cmath>
#include "Macros.h"

class Profiler {
public:
    struct RegionData {
        double totalMs = 0;
        uint64_t count = 0;
    };

    // --- RAII ---

    struct ScopedTimer {
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        Profiler& parent;
        ScopedTimer(Profiler& p) : parent(p), start(std::chrono::high_resolution_clock::now()) {}
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            parent.addFrame(ms);
        }
    };

    struct RegionTimer {
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::string tag;
        RegionTimer(std::string name) : tag(std::move(name)), start(std::chrono::high_resolution_clock::now()) {}
        ~RegionTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            Profiler::recordRegion(tag, ms);
        }
    };

    Profiler(int targetTotalLoops = 12000, int warmUpFrames = 2000)
        : _targetTotalLoops(targetTotalLoops), _warmUpFrames(warmUpFrames) {
        _frameTimes.reserve(targetTotalLoops + warmUpFrames);
    }

    [[nodiscard]] ScopedTimer scope() {
        _currentLoopTimes++;
        return ScopedTimer(*this);
    }

    static void recordRegion(const std::string& name, double ms) {
        auto& data = _regionRegistry[name];
        data.totalMs += ms;
        data.count++;
    }

    void addFrame(double ms) { _frameTimes.push_back(ms); }
    bool needToLoop() const { return _currentLoopTimes < _targetTotalLoops; }

    void printReport(const std::string& configName) {
        if (_frameTimes.size() <= _warmUpFrames) {
            std::cout << "Error! Data is not enough to output" << std::endl;
            return;
        }

        auto validStart = _frameTimes.begin() + _warmUpFrames;
        auto validEnd = _frameTimes.end();
        size_t validCount = std::distance(validStart, validEnd);

        double mean = 0.0, m2 = 0.0;
        for (size_t i = 0; i < validCount; ++i) {
            double x = *(validStart + i);
            double delta = x - mean;
            mean += delta / (i + 1);
            m2 += delta * (x - mean);
        }
        double stdDev = std::sqrt(m2 / validCount);

        // P99
        std::vector<double> tmp(validStart, validEnd);
        size_t p99Idx = static_cast<size_t>(validCount * 0.99);
        std::nth_element(tmp.begin(), tmp.begin() + p99Idx, tmp.end());
        double p99Ms = tmp[p99Idx];

		// Main Report
        std::cout << std::format("\n[ PERFORMANCE REPORT: {} ]\n", configName);
        std::cout << std::format("{:<20}: {} frames\n", "Sample Count", validCount);
        std::cout << std::format("{:<20}: {:.2f} FPS\n", "Average FPS", 1000.0 / mean);
        std::cout << std::format("{:<20}: {:.2f} ms\n", "Avg Frame Time", mean);
        std::cout << std::format("{:<20}: {:.2f} ms\n", "P99 (Worst Case)", p99Ms);
        std::cout << std::format("{:<20}: {:.2f} ms\n", "Std Dev (Jitter)", stdDev);

		// Region Breakdown
        if (!_regionRegistry.empty()) {
            std::cout << "\n[ Region Breakdown (Avg ms/frame) ]\n";
            for (auto const& [name, data] : _regionRegistry) {
                std::cout << std::format("{:<20}: {:.4f} ms\n", name, data.totalMs / validCount);
            }
        }
        std::cout << "------------------------------------------\n";

        clear();
    }

    void clear() {
        _frameTimes.clear();
        _regionRegistry.clear();
        _currentLoopTimes = 0;
    }

private:
    std::vector<double> _frameTimes;

    static inline std::map<std::string, RegionData> _regionRegistry;
    int _currentLoopTimes = 0;
    int _targetTotalLoops, _warmUpFrames;
};