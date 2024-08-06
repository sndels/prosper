
#ifndef PROSPER_SCENE_ANIMATIONS_HPP
#define PROSPER_SCENE_ANIMATIONS_HPP

#include "Allocators.hpp"
#include "scene/Accessors.hpp"

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <wheels/containers/array.hpp>

enum class InterpolationType
{
    Step,
    Linear,
    CubicSpline
};

template <typename T> class Animation
{
  public:
    Animation(
        InterpolationType interpolation, TimeAccessor &&timeFrames,
        ValueAccessor<T> &&valueFrames);

    [[nodiscard]] float endTimeS() const;

    void registerTarget(T &target);
    void update(float timeS);

  private:
    // General because we don't know how many of these we'll have beforehand
    // TODO: tlsf and linear allocators for diffrent world use cases?
    wheels::Array<T *> m_targets{gAllocators.general};

    InterpolationType m_interpolation{InterpolationType::Step};
    TimeAccessor m_timeFrames;
    ValueAccessor<T> m_valueFrames;
};

struct Animations
{
    wheels::Array<Animation<glm::vec3>> vec3{gAllocators.world};
    wheels::Array<Animation<glm::quat>> quat{gAllocators.world};
};

template <typename T>
Animation<T>::Animation(
    InterpolationType interpolation, TimeAccessor &&timeFrames,
    ValueAccessor<T> &&valueFrames)
: m_interpolation{interpolation}
, m_timeFrames{WHEELS_FWD(timeFrames)}
, m_valueFrames{WHEELS_FWD(valueFrames)}
{
}

template <typename T> void Animation<T>::registerTarget(T &target)
{
    m_targets.push_back(&target);
}

template <typename T> float Animation<T>::endTimeS() const
{
    return m_timeFrames.endTimeS();
}

template <typename T> void Animation<T>::update(float timeS)
{
    const KeyFrameInterpolation interp = m_timeFrames.interpolation(timeS);

    T firstValue{};
    if (interp.t == 0.f && m_interpolation == InterpolationType::CubicSpline)
        // Three values per frame, property value is the middle one
        firstValue = m_valueFrames.read(interp.firstFrame * 3 + 1);
    else
        firstValue = m_valueFrames.read(interp.firstFrame);

    T value{};
    if (interp.t == 0.f || m_interpolation == InterpolationType::Step)
        value = firstValue;
    else
    {
        if (m_interpolation == InterpolationType::Linear)
        {
            const T secondValue = m_valueFrames.read(interp.firstFrame + 1);
            if constexpr (wheels::SameAs<T, glm::quat>)
                value = glm::slerp(firstValue, secondValue, interp.t);
            else
                value = (1.f - interp.t) * firstValue + interp.t * secondValue;
        }
        else if (m_interpolation == InterpolationType::CubicSpline)
        {
            // Three values per keyframe: in-tangent, property, out-tangent
            const uint32_t firstIndex = interp.firstFrame * 3;
            const T vk = m_valueFrames.read(firstIndex + 1);
            const T bk = m_valueFrames.read(firstIndex + 2);
            const T vk1 = m_valueFrames.read(firstIndex + 3 + 1);
            const T ak1 = m_valueFrames.read(firstIndex + 3);

            const float t3 = interp.t * interp.t * interp.t;
            const float t2 = interp.t * interp.t;
            const float t2Times2 = 2.f * interp.t * interp.t;
            const float t3Times2 = 2.f * interp.t * interp.t * interp.t;
            const float t2Times3 = 3.f * interp.t * interp.t;
            const float td = interp.stepDuration;
            const float t = interp.t;

            value = (t3Times2 - t2Times3 + 1.f) * vk +
                    td * (t3 - t2Times2 + t) * bk +
                    (-t3Times2 + t2Times3) * vk1 + td * (t3 - t2) * ak1;

            if constexpr (wheels::SameAs<T, glm::quat>)
                value = glm::normalize(value);
        }
        else
            WHEELS_ASSERT(!"Unimplemented interpolation mode");
    }

    for (T *target : m_targets)
        *target = value;
}

#endif // PROSPER_SCENE_ANIMATIONS_HPP
