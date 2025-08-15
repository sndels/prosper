#ifndef PROSPER_UTILS_TIMER_HPP
#define PROSPER_UTILS_TIMER_HPP

#include <chrono>

namespace utils
{

class Timer
{
  public:
    Timer() noexcept;

    void reset();
    [[nodiscard]] float getSeconds() const;

  private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
};

} // namespace utils

#endif // PROSPER_UTILS_TIMER_HPP
