#ifndef GBUFFER_PC_GLSL
#define GBUFFER_PC_GLSL

layout(push_constant) uniform GBufferPC
{
    // Some of these are mirrored between this and ForwardPC
    uint ModelInstanceID;
    uint MeshID;
    uint MaterialID;
    uint previousTransformValid;
}
PC;

#endif // GBUFFER_PC_GLSL
