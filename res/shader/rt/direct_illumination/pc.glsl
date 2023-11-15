#ifndef RT_DIRECT_ILLUMINATION_PC_GLSL
#define RT_DIRECT_ILLUMINATION_PC_GLSL

layout(push_constant) uniform RtDirectIlluminationPC
{
    uint drawType;
    uint frameIndex;
}
PC;

#endif // RT_DIRECT_ILLUMINATION_PC_GLSL
