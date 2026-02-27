#ifndef PARTICLES_PARTICLE_H
#define PARTICLES_PARTICLE_H

#include "../fields.h"

#ifdef __cplusplus
namespace particles::shader_structs
{
#endif // __cplusplus

#ifndef __cplusplus

const uint ParticleMaskBits_Gravity = 1 << 0;

bool gravityEnabled(uint mask) { return bitfieldExtract(mask, 0, 1) == 1; }

#endif // __cplusplus

struct Particle
{
    STRUCT_FIELD_GLM(vec4, position_lifetime, {-9999.f});
    STRUCT_FIELD_GLM(vec4, normal_spawnRate, {});
    STRUCT_FIELD_GLM(vec4, velocity_spawnCounter, {});
    STRUCT_FIELD(float, childLifetime, 0.f);
    STRUCT_FIELD(float, childVelocity, 0.f);
    STRUCT_FIELD_GLM(uint, mask, 0);
    STRUCT_FIELD_GLM(uint, _pad0, 0);
};

#ifdef __cplusplus

// Tight packing is assumed
static_assert(alignof(Particle) == sizeof(uint32_t));
static_assert((sizeof(Particle) % 16) == 0);

} // namespace particles::shader_structs

#endif // __cplusplus

#endif // PARTICLES_PARTICLE_H
