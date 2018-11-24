#ifndef UTIL_TIMER_H_
#define UTIL_TIMER_H_

#include <chrono>

class Timer
{
private:
    std::chrono::high_resolution_clock::time_point start_;

public:
    Timer(bool run = true)
    {
        if (run)
        {
            reset();
        }
    }

    void reset()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    int64_t elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_).count();
    }
};

#endif