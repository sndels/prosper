#ifndef PAYLOAD_GLSL
#define PAYLOAD_GLSL

struct RayPayload
{
    uint instanceCustomIndex;
    uint primitiveID;
    vec2 baryCoord;
    uint randomSeed;
};

#endif // PAYLOAD_GLSL
