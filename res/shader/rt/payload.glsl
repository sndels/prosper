#ifndef RT_PAYLOAD_GLSL
#define RT_PAYLOAD_GLSL

struct RayPayload
{
    uint drawInstanceIndex;
    uint primitiveID;
    vec2 baryCoord;
    uint randomSeed;
};

#endif // RT_PAYLOAD_GLSL
