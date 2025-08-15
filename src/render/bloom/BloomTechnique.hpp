
#ifndef PROSPER_RENDER_BLOOM_BLOOM_TECHNIQUE_HPP
#define PROSPER_RENDER_BLOOM_BLOOM_TECHNIQUE_HPP

#include <cstdint>
#include <wheels/containers/static_array.hpp>

#define BLOOM_TECHNIQUES MultiResolutionBlur, Fft
#define BLOOM_TECHNIQUES_AND_COUNT BLOOM_TECHNIQUES, Count

namespace render::bloom
{

enum class BloomTechnique : uint8_t
{
    BLOOM_TECHNIQUES_AND_COUNT
};

extern const wheels::StaticArray<
    const char *, static_cast<size_t>(BloomTechnique::Count)>
    sBloomTechniqueNames;

} // namespace render::bloom

#endif // PROSPER_RENDER_BLOOM_BLOOM_TECHNIQUE_HPP
