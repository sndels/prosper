#include "timer.hpp"

Timer::Timer() : _start(std::chrono::system_clock::now()) {}

void Timer::reset() { _start = std::chrono::system_clock::now(); }

float Timer::getSeconds() const {
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> dt = end - _start;
    return dt.count();
}
