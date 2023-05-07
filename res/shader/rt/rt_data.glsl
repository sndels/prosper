#ifndef RT_RT_DATA_GLSL
#define RT_RT_DATA_GLSL

layout(set = RAY_TRACING_SET, binding = 0) uniform accelerationStructureEXT as;
struct RTInstance
{
    uint modelInstanceID;
    uint meshID;
    uint materialID;
};
layout(std430, set = RAY_TRACING_SET, binding = 1) readonly buffer RTInstances
{
    RTInstance data[];
}
rtInstances;

#endif // RT_RT_DATA_GLSL
