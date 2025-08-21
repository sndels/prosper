#include "ResolutionScale.hpp"

#include "utils/ForEach.hpp"

#define BLOOM_RESOLUTION_SCALE_TYPES_STRINGIFY(t) #t,

#define BLOOM_RESOLUTION_SCALE_TYPES_STRS                                      \
    FOR_EACH(                                                                  \
        BLOOM_RESOLUTION_SCALE_TYPES_STRINGIFY, BLOOM_RESOLUTION_SCALE_TYPES)

namespace render::bloom
{

const wheels::StaticArray<
    const char *, static_cast<size_t>(ResolutionScale::Count)>
    sResolutionScaleTypeNames{{BLOOM_RESOLUTION_SCALE_TYPES_STRS}};

uint32_t bloomResolutionScale(ResolutionScale scale)
{
    return (static_cast<uint32_t>(scale) + 1) * 2;
}

} // namespace render::bloom
