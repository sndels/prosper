#ifndef FORWARD_PC_GLSL
#define FORWARD_PC_GLSL

layout(push_constant) uniform ForwardPC
{
    // Some of these are mirrored between this and GBufferPC
    uint ModelInstanceID;
    uint MeshID;
    uint MaterialID;
    uint DrawType;
    uint ibl;
}
PC;

#endif // FORWARD_PC_GLSL
