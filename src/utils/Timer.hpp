#ifndef PROSPER_UTILS_TIMER_HPP
#define PROSPER_UTILS_TIMER_HPP

#include <chrono>

class Timer
{
  public:
    Timer();

    void reset();
    [[nodiscard]] float getSeconds() const;

  private:
    std::chrono::time_point<std::chrono::system_clock> _start;
};

#endif // PROSPER_UTILS_TIMER_HPP
