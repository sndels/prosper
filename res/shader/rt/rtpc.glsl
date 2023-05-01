#ifndef RTPC_GLSL
#define RTPC_GLSL

layout(push_constant) uniform RtPC
{
    uint drawType;
    uint flags;
    uint frameIndex;
}
rtPC;

bool flagColorDirty() { return bitfieldExtract(rtPC.flags, 0, 1) == 1; }

bool flagAccumulate() { return bitfieldExtract(rtPC.flags, 1, 1) == 1; }
bool flagIBL() { return bitfieldExtract(rtPC.flags, 2, 1) == 1; }

#endif // RTPC_GLSL
