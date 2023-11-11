#ifndef RT_RTPC_GLSL
#define RT_RTPC_GLSL

layout(push_constant) uniform RtPC
{
    uint drawType;
    uint flags;
    uint frameIndex;
    float apertureDiameter;
    float focusDistance;
    float focalLength;
    uint rouletteStartBounce;
}
rtPC;

bool flagSkipHistory() { return bitfieldExtract(rtPC.flags, 0, 1) == 1; }
bool flagAccumulate() { return bitfieldExtract(rtPC.flags, 1, 1) == 1; }
bool flagIBL() { return bitfieldExtract(rtPC.flags, 2, 1) == 1; }
bool flagDepthOfField() { return bitfieldExtract(rtPC.flags, 3, 1) == 1; }
bool flagClampIndirect() { return bitfieldExtract(rtPC.flags, 4, 1) == 1; }

#endif // RT_RTPC_GLSL
