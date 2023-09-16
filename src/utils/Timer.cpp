#include "Timer.hpp"

Timer::Timer()
: _start(std::chrono::system_clock::now())
{
}

void Timer::reset() { _start = std::chrono::system_clock::now(); }

float Timer::getSeconds() const
{
    const auto end = std::chrono::system_clock::now();
    const std::chrono::duration<float> dt = end - _start;
    return dt.count();
}
