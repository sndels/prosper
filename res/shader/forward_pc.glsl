#ifndef FORWARD_PC_GLSL
#define FORWARD_PC_GLSL

layout(push_constant) uniform ForwardPC
{
    // Some of these are mirrored between this and GBufferPC
    uint DrawInstanceID;
    uint DrawType;
    uint ibl;
    uint previousTransformValid;
}
PC;

#endif // FORWARD_PC_GLSL
