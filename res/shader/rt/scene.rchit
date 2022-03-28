#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#pragma shader_stage(closest)

#include "rt/payload.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    payload.instanceCustomIndex = gl_InstanceCustomIndexEXT;
    payload.primitiveID = gl_PrimitiveID;
}
