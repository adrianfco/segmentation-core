#pragma once

#include <cstdint>

// Hardware counters read around the timed region, so they cover the same work
// as the timer. Linux only, and everything falls back to "unavailable" when the
// kernel refuses the events: that is the case in CI, and on any machine where
// kernel.perf_event_paranoid is above 2.
//
// The counters are opened on the calling thread. OpenMP workers already exist
// when the region starts, so they are not inherited and multi-threaded runs
// report nothing.

namespace bench {

struct CounterValues {
    bool               valid           = false;
    unsigned long long cycles          = 0;
    unsigned long long instructions    = 0;
    unsigned long long branch_misses   = 0;
    unsigned long long l1d_read_misses = 0;
};

} // namespace bench

#if defined(__linux__)

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>

namespace bench {

class Counters {
public:
    Counters() {
        struct Event { std::uint32_t type; std::uint64_t config; };
        // Generic LLC events are unmapped on Zen, and the per-core fill-source
        // events credit prefetched lines to L2, so there is no trustworthy
        // memory-traffic counter to open here.
        const Event events[kCount] = {
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
            {PERF_TYPE_HW_CACHE,  PERF_COUNT_HW_CACHE_L1D |
                                 (PERF_COUNT_HW_CACHE_OP_READ     << 8) |
                                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
        };

        for (int i = 0; i < kCount; ++i) {
            fds_[i] = open_event(events[i].type, events[i].config, (i == 0) ? -1 : fds_[0]);
            if (fds_[i] < 0) { close_all(); return; }
        }
    }

    ~Counters() { close_all(); }

    Counters(const Counters&)            = delete;
    Counters& operator=(const Counters&) = delete;

    bool available() const { return fds_[0] >= 0; }

    void start() {
        if (!available()) return;
        ioctl(fds_[0], PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
        ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    void stop() {
        if (!available()) return;
        ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }

    CounterValues read() const {
        CounterValues v;
        if (!available()) return v;

        // PERF_FORMAT_GROUP lays the read out as nr, time_enabled, time_running,
        // then one value per event in the order they were opened.
        std::uint64_t buf[3 + kCount] = {};
        if (::read(fds_[0], buf, sizeof(buf)) != static_cast<ssize_t>(sizeof(buf))) return v;
        // time_enabled != time_running means the group was multiplexed, so the
        // counts are scaled estimates rather than exact
        if (buf[0] != kCount || buf[1] != buf[2]) return v;

        v.valid           = true;
        v.cycles          = buf[3];
        v.instructions    = buf[4];
        v.branch_misses   = buf[5];
        v.l1d_read_misses = buf[6];
        return v;
    }

private:
    static constexpr int kCount = 4;

    static int open_event(std::uint32_t type, std::uint64_t config, int group) {
        perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.size           = sizeof(attr);
        attr.type           = type;
        attr.config         = config;
        attr.disabled       = (group == -1) ? 1 : 0;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        attr.read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                              PERF_FORMAT_TOTAL_TIME_RUNNING;
        return static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, group, 0));
    }

    void close_all() {
        for (int i = kCount - 1; i >= 0; --i) {
            if (fds_[i] >= 0) close(fds_[i]);
            fds_[i] = -1;
        }
    }

    int fds_[kCount] = {-1, -1, -1, -1};
};

} // namespace bench

#else

namespace bench {

class Counters {
public:
    bool available() const { return false; }
    void start() {}
    void stop() {}
    CounterValues read() const { return {}; }
};

} // namespace bench

#endif
