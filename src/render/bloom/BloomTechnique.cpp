#include "BloomTechnique.hpp"

#include "utils/ForEach.hpp"

#define BLOOM_TECHNIQUES_STRINGIFY(t) #t,
#define BLOOM_TECHNIQUE_STRS                                                   \
    FOR_EACH(BLOOM_TECHNIQUES_STRINGIFY, BLOOM_TECHNIQUES)

namespace render::bloom
{

const wheels::StaticArray<
    const char *, static_cast<size_t>(BloomTechnique::Count)>
    sBloomTechniqueNames{{BLOOM_TECHNIQUE_STRS}};

} // namespace render::bloom
