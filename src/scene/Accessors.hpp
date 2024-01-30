#ifndef PROSPER_SCENE_ACCESSORS_HPP
#define PROSPER_SCENE_ACCESSORS_HPP

#include <cstdint>

struct KeyFrameInterpolation
{
    // If t == 0.f, use start frame directly
    // float comparison is sus but it's guaranteed to work when we store 0.f
    // explicitly
    float t{0.f};
    float stepDuration{0.f};
    uint32_t firstFrame{0};
};

class TimeAccessor
{
  public:
    struct Interval
    {
        float startTimeS{0.f};
        float endTimeS{0.f};
    };
    TimeAccessor(
        const float *data, uint32_t count, const Interval &interval) noexcept;

    float endTimeS() const;
    KeyFrameInterpolation interpolation(float timeS) const;

  private:
    const float *_data{nullptr};
    uint32_t _count{0};
    Interval _interval;
};

// Templated so that Sampler can be templated on the read value type.
// Concrete implementations for supported types
template <typename T> class ValueAccessor
{
  public:
    // Count is for vector elements, not individual float
    ValueAccessor(const uint8_t *data, uint32_t count) noexcept;

    T read(uint32_t index) const;

  private:
    const uint8_t *_data{nullptr};
    uint32_t _count{0};
};
// TODO:
// Requires or something to give an error for unimplemented value types

#endif // PROSPER_SCENE_ACCESSORS_HPP
