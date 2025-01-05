
#ifndef PROSPER_RENDER_BLOOM_RESOLUTION_SCALE_HPP
#define PROSPER_RENDER_BLOOM_RESOLUTION_SCALE_HPP

#include <cstdint>
#include <wheels/containers/static_array.hpp>

#define BLOOM_RESOLUTION_SCALE_TYPES Half, Quarter

#define BLOOM_RESOLUTION_SCALE_TYPES_AND_COUNT                                 \
    BLOOM_RESOLUTION_SCALE_TYPES, Count

// NOLINTNEXTLINE(performance-enum-size) specialization constant
enum class BloomResolutionScale : uint32_t
{
    BLOOM_RESOLUTION_SCALE_TYPES_AND_COUNT
};

uint32_t bloomResolutionScale(BloomResolutionScale scale);

extern const wheels::StaticArray<
    const char *, static_cast<size_t>(BloomResolutionScale::Count)>
    sResolutionScaleTypeNames;

#endif // PROSPER_RENDER_BLOOM_RESOLUTION_SCALE_HPP
