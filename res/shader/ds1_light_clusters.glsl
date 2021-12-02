// This needs to match engine
#define LIGHT_CLUSTER_DIMENSION 64

#ifdef WRITE_CULLING_BINDS
#define UNIFORM_TYPE writeonly
#define COUNT_UNIFORM_TYPE
#else
#define COUNT_UNIFORM_TYPE readonly
#define UNIFORM_TYPE readonly
#endif

layout(set = 1, binding = 0, rg32ui) uniform UNIFORM_TYPE uimage2D
    clusterPointers;
layout(set = 1, binding = 1, r32ui) uniform COUNT_UNIFORM_TYPE uimageBuffer
    lightIndicesCount;
layout(set = 1, binding = 2, r16ui) uniform UNIFORM_TYPE uimageBuffer
    lightIndices;

#ifdef WRITE_CULLING_BINDS
uvec4 packClusterPointer(uint indexOffset, uint pointCount, uint spotCount)
{
    return uvec4(indexOffset, (pointCount << 16) | spotCount, 0, 0);
}
#else
void unpackClusterPointer(
    uvec2 index, out uint indexOffset, out uint pointCount, out uint spotCount)
{
    uvec2 packed = imageLoad(clusterPointers, ivec2(index)).xy;
    indexOffset = packed.x;
    pointCount = packed.y >> 16;
    spotCount = packed.y & 0xFFFF;
}
#endif // WRITE_CULLING_BINDS
