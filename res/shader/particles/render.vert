#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

#include "../scene/camera.glsl"
#include "../shared/shader_structs/particles/particle.h"

layout(std430, set = PARTICLES_SET, binding = 0) buffer readonly InParticles
{
    Particle particles[];
}
inParticles;

layout(location = 0) out vec3 fragTexCoord;
layout(location = 1) out vec4 fragPositionNDC;
layout(location = 2) out vec4 fragPrevPositionNDC;

void main()
{
    uint particleIndex = gl_InstanceIndex;
    uint vertexIndex = gl_VertexIndex;

    vec4 positionLifetime =
        inParticles.particles[particleIndex].position_lifetime;
    vec3 positionWorld = positionLifetime.xyz;
    if (positionLifetime.w < 0.)
        // No discard so let's abuse clipping
        // TODO: Compaction step for live particles?
        positionWorld = vec3(1. / 0.);

    // Output topology is a triangle strip
    // 2 ----- 3
    // | \     |
    // |   \   |
    // |     \ |
    // 0 ----- 1
    float xOffset = (2. * (vertexIndex % 2) - 1.) * .001;
    float yOffset = -(2. * (vertexIndex / 2) - 1.) * .001;
    positionWorld += normalize(cameraWorldUp()) * yOffset;
    positionWorld += normalize(cameraWorldRight()) * xOffset;

    vec4 positionClip =
        camera.cameraToClip * camera.worldToCamera * vec4(positionWorld, 1);
    gl_Position = positionClip;
}
