#extension GL_EXT_ray_tracing : require

#pragma shader_stage(closest)

#include "payload.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 baryCoord;

void main()
{
    payload.drawInstanceIndex = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    payload.primitiveID = gl_PrimitiveID;
    payload.baryCoord = baryCoord;
}
