#ifndef RT_DIRECT_ILLUMINATION_PC_GLSL
#define RT_DIRECT_ILLUMINATION_PC_GLSL

layout(push_constant) uniform RtDirectIlluminationPC
{
    uint drawType;
    uint frameIndex;
    uint flags;
}
PC;

bool flagSkipHistory() { return bitfieldExtract(PC.flags, 0, 1) == 1; }
bool flagAccumulate() { return bitfieldExtract(PC.flags, 1, 1) == 1; }

#endif // RT_DIRECT_ILLUMINATION_PC_GLSL
