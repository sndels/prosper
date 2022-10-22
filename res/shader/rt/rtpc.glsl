#ifndef RTPC_GLSL
#define RTPC_GLSL

layout(push_constant) uniform RtPC { uint drawType; }
rtPC;

const uint DrawType_PrimitiveID = 0;
const uint DrawType_MeshID = 1;
const uint DrawType_MaterialID = 2;

#endif // RTPC_GLSL
