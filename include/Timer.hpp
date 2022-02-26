#ifndef TIMER_HPP
#define TIMER_HPP

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

#endif // TIMER_HPP
