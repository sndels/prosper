#include "Accessors.hpp"

#include "../utils/Utils.hpp"
#include <glm/gtc/quaternion.hpp>

using namespace glm;

TimeAccessor::TimeAccessor(
    const float *data, uint32_t count, const Interval &interval)
: _data{data}
, _count{count}
, _interval{interval}
{
    WHEELS_ASSERT(_data != nullptr);
    WHEELS_ASSERT(_count > 0);
}

float TimeAccessor::endTimeS() const { return _interval.endTimeS; }

KeyFrameInterpolation TimeAccessor::interpolation(float timeS) const
{
    if (timeS <= _interval.startTimeS || timeS < _data[0])
        return KeyFrameInterpolation{
            .t = 0.f,
            .firstFrame = 0,
        };

    if (timeS >= _interval.endTimeS)
        return KeyFrameInterpolation{
            .t = 0.f,
            .firstFrame = _count - 1,
        };

    // TODO:
    // Cache previous first frame to speed up consequtive queries. Average case
    // is within the same interval, next one if not. Random access still
    // requires search. Need to be careful that the cache check works with
    // random access. Profile well.

    KeyFrameInterpolation ret;
    const uint32_t lastFrame = _count - 1;
    while (ret.firstFrame < lastFrame)
    {
        const float frameTimeS = _data[ret.firstFrame];
        // Time == first frame has an early out
        if (frameTimeS > timeS)
            break;

        ret.firstFrame++;
    }
    // We went one over in the loop
    ret.firstFrame--;

    if (ret.firstFrame < lastFrame)
    {
        const float firstTime = _data[ret.firstFrame];
        const float secondTime = _data[ret.firstFrame + 1];
        WHEELS_ASSERT(firstTime <= timeS);
        WHEELS_ASSERT(timeS <= secondTime);

        const float duration = secondTime - firstTime;
        ret.stepDuration = timeS - firstTime;
        ret.t = ret.stepDuration / duration;
    }
    // else, firstFrame == lastFrame and t == 0.f signals we should clamp
    WHEELS_ASSERT(ret.t >= 0.f);
    WHEELS_ASSERT(ret.t <= 1.f);

    return ret;
}

template <>
ValueAccessor<vec3>::ValueAccessor(const uint8_t *data, uint32_t count)
: _data{data}
, _count{count}
{
    WHEELS_ASSERT(_data != nullptr);
    WHEELS_ASSERT(_count > 0);
}

template <> vec3 ValueAccessor<vec3>::read(uint32_t index) const
{
    WHEELS_ASSERT(index < _count);
    return *reinterpret_cast<const vec3 *>(
        _data + asserted_cast<size_t>(index) * 3 * sizeof(float));
}

template <>
ValueAccessor<quat>::ValueAccessor(const uint8_t *data, uint32_t count)
: _data{data}
, _count{count}
{
    WHEELS_ASSERT(_data != nullptr);
    WHEELS_ASSERT(_count > 0);
}

template <> quat ValueAccessor<quat>::read(uint32_t index) const
{
    WHEELS_ASSERT(index < _count);
    return *reinterpret_cast<const quat *>(
        _data + asserted_cast<size_t>(index) * 4 * sizeof(float));
}
