#ifndef RT_REFERENCE_RTPC_GLSL
#define RT_REFERENCE_RTPC_GLSL

layout(push_constant) uniform ReferenceRtPC
{
    uint drawType;
    uint flags;
    uint frameIndex;
    float apertureDiameter;
    float focusDistance;
    float focalLength;
    uint rouletteStartBounce;
    uint maxBounces;
}
PC;

bool flagSkipHistory() { return bitfieldExtract(PC.flags, 0, 1) == 1; }
bool flagAccumulate() { return bitfieldExtract(PC.flags, 1, 1) == 1; }
bool flagIBL() { return bitfieldExtract(PC.flags, 2, 1) == 1; }
bool flagDepthOfField() { return bitfieldExtract(PC.flags, 3, 1) == 1; }
bool flagClampIndirect() { return bitfieldExtract(PC.flags, 4, 1) == 1; }

#endif // RT_REFERENCE_RTPC_GLSL
