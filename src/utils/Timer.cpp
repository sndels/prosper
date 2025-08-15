#include "Timer.hpp"

namespace utils
{

Timer::Timer() noexcept
: m_start(std::chrono::system_clock::now())
{
}

void Timer::reset() { m_start = std::chrono::system_clock::now(); }

float Timer::getSeconds() const
{
    const auto end = std::chrono::system_clock::now();
    const std::chrono::duration<float> dt = end - m_start;
    return dt.count();
}

} // namespace utils
