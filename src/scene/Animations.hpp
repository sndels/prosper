
#ifndef PROSPER_SCENE_ANIMATIONS_HPP
#define PROSPER_SCENE_ANIMATIONS_HPP

#include "../Allocators.hpp"
#include "Accessors.hpp"

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

    float endTimeS() const;

    void registerTarget(T &target);
    void update(float timeS);

  private:
    // General because we don't know how many of these we'll have beforehand
    // TODO: tlsf and linear allocators for diffrent world use cases?
    wheels::Array<T *> _targets{gAllocators.general};

    InterpolationType _interpolation{InterpolationType::Step};
    TimeAccessor _timeFrames;
    ValueAccessor<T> _valueFrames;
};

struct Animations
{
    wheels::Array<Animation<glm::vec3>> _vec3{gAllocators.world};
    wheels::Array<Animation<glm::quat>> _quat{gAllocators.world};
};

template <typename T>
Animation<T>::Animation(
    InterpolationType interpolation, TimeAccessor &&timeFrames,
    ValueAccessor<T> &&valueFrames)
: _interpolation{interpolation}
, _timeFrames{WHEELS_FWD(timeFrames)}
, _valueFrames{WHEELS_FWD(valueFrames)}
{
}

template <typename T> void Animation<T>::registerTarget(T &target)
{
    _targets.push_back(&target);
}

template <typename T> float Animation<T>::endTimeS() const
{
    return _timeFrames.endTimeS();
}

template <typename T> void Animation<T>::update(float timeS)
{
    const KeyFrameInterpolation interp = _timeFrames.interpolation(timeS);

    T firstValue;
    if (interp.t == 0.f && _interpolation == InterpolationType::CubicSpline)
        // Three values per frame, property value is the middle one
        firstValue = _valueFrames.read(interp.firstFrame * 3 + 1);
    else
        firstValue = _valueFrames.read(interp.firstFrame);

    T value{};
    if (interp.t == 0.f || _interpolation == InterpolationType::Step)
        value = firstValue;
    else
    {
        if (_interpolation == InterpolationType::Linear)
        {
            const T secondValue = _valueFrames.read(interp.firstFrame + 1);
            if constexpr (wheels::SameAs<T, glm::quat>)
                value = glm::slerp(firstValue, secondValue, interp.t);
            else
                value = (1.f - interp.t) * firstValue + interp.t * secondValue;
        }
        else if (_interpolation == InterpolationType::CubicSpline)
        {
            // Three values per keyframe: in-tangent, property, out-tangent
            const uint32_t firstIndex = interp.firstFrame * 3;
            const T vk = _valueFrames.read(firstIndex + 1);
            const T bk = _valueFrames.read(firstIndex + 2);
            const T vk1 = _valueFrames.read(firstIndex + 3 + 1);
            const T ak1 = _valueFrames.read(firstIndex + 3);

            const float t3 = interp.t * interp.t * interp.t;
            const float t2 = interp.t * interp.t;
            const float _2t2 = 2.f * interp.t * interp.t;
            const float _2t3 = 2.f * interp.t * interp.t * interp.t;
            const float _3t2 = 3.f * interp.t * interp.t;
            const float td = interp.stepDuration;
            const float t = interp.t;

            value = (_2t3 - _3t2 + 1.f) * vk + td * (t3 - _2t2 + t) * bk +
                    (-_2t3 + _3t2) * vk1 + td * (t3 - t2) * ak1;

            if constexpr (wheels::SameAs<T, glm::quat>)
                value = glm::normalize(value);
        }
        else
            WHEELS_ASSERT(!"Unimplemented interpolation mode");
    }

    for (T *target : _targets)
        *target = value;
}

#endif // PROSPER_SCENE_ANIMATIONS_HPP
