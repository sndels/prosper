#ifndef PARTICLES_PARTICLE_H
#define PARTICLES_PARTICLE_H

#include "../fields.h"

#ifdef __cplusplus
namespace particles::shader_structs
{
#endif // __cplusplus

#ifndef __cplusplus

const uint ParticleMaskBits_Gravity = 1 << 0;
const uint ParticleMaskBits_Decay = 1 << 1;
const uint ParticleMaskBits_Emit = 1 << 2;

bool gravityEnabled(uint mask) { return bitfieldExtract(mask, 0, 1) == 1; }
bool decayEnabled(uint mask) { return bitfieldExtract(mask, 1, 1) == 1; }
bool emitEnabled(uint mask) { return bitfieldExtract(mask, 2, 1) == 1; }

#endif // __cplusplus

struct Particle
{
    STRUCT_FIELD_GLM(vec4, position_lifetime, {-9999.f});
    STRUCT_FIELD_GLM(vec4, normal_spawnRateS, {});
    STRUCT_FIELD_GLM(vec4, velocity_spawnTimerS, {});
    STRUCT_FIELD_GLM(uint, mask, 0);
    STRUCT_FIELD_GLM(uint, _pad0, 0);
    STRUCT_FIELD_GLM(uint, _pad1, 0);
    STRUCT_FIELD_GLM(uint, _pad2, 0);
};

#ifdef __cplusplus

// Tight packing is assumed
static_assert(alignof(Particle) == sizeof(uint32_t));
static_assert((sizeof(Particle) % 16) == 0);

} // namespace particles::shader_structs

#endif // __cplusplus

#endif // PARTICLES_PARTICLE_H
