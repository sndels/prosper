#ifndef FORWARD_PC_GLSL
#define FORWARD_PC_GLSL

layout(push_constant) uniform ForwardPC
{
    uint ModelInstanceID;
    uint MeshID;
    uint MaterialID;
    uint DrawType;
}
forwardPC;

#endif // FORWARD_PC_GLSL
