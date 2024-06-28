#include "Accessors.hpp"

#include "../utils/Utils.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace glm;

TimeAccessor::TimeAccessor(
    const float *data, uint32_t count, const Interval &interval) noexcept
: m_data{data}
, m_count{count}
, m_interval{interval}
{
    WHEELS_ASSERT(m_data != nullptr);
    WHEELS_ASSERT(m_count > 0);
}

float TimeAccessor::endTimeS() const { return m_interval.endTimeS; }

KeyFrameInterpolation TimeAccessor::interpolation(float timeS) const
{
    if (timeS <= m_interval.startTimeS || timeS < m_data[0])
        return KeyFrameInterpolation{
            .t = 0.f,
            .firstFrame = 0,
        };

    if (timeS >= m_interval.endTimeS)
        return KeyFrameInterpolation{
            .t = 0.f,
            .firstFrame = m_count - 1,
        };

    // TODO:
    // Cache previous first frame to speed up consequtive queries. Average case
    // is within the same interval, next one if not. Random access still
    // requires search. Need to be careful that the cache check works with
    // random access. Profile well.

    KeyFrameInterpolation ret;
    const uint32_t lastFrame = m_count - 1;
    while (ret.firstFrame < lastFrame)
    {
        const float frameTimeS = m_data[ret.firstFrame];
        // Time == first frame has an early out
        if (frameTimeS > timeS)
            break;

        ret.firstFrame++;
    }
    // We went one over in the loop
    ret.firstFrame--;

    if (ret.firstFrame < lastFrame)
    {
        const float firstTime = m_data[ret.firstFrame];
        const float secondTime = m_data[ret.firstFrame + 1];
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
ValueAccessor<vec3>::ValueAccessor(const uint8_t *data, uint32_t count) noexcept
: m_data{data}
, m_count{count}
{
    WHEELS_ASSERT(m_data != nullptr);
    WHEELS_ASSERT(m_count > 0);
}

template <> vec3 ValueAccessor<vec3>::read(uint32_t index) const
{
    WHEELS_ASSERT(index < m_count);
    return *reinterpret_cast<const vec3 *>(
        m_data + asserted_cast<size_t>(index) * 3 * sizeof(float));
}

template <>
ValueAccessor<quat>::ValueAccessor(const uint8_t *data, uint32_t count) noexcept
: m_data{data}
, m_count{count}
{
    WHEELS_ASSERT(m_data != nullptr);
    WHEELS_ASSERT(m_count > 0);
}

template <> quat ValueAccessor<quat>::read(uint32_t index) const
{
    WHEELS_ASSERT(index < m_count);
    return *reinterpret_cast<const quat *>(
        m_data + asserted_cast<size_t>(index) * 4 * sizeof(float));
}
