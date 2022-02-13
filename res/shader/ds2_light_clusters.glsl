// These needs to match engine
#define LIGHT_CLUSTER_DIMENSION 32
#define LIGHT_CLUSTER_Z_SLICE_COUNT 16

#ifdef WRITE_CULLING_BINDS
#define UNIFORM_TYPE writeonly
#define COUNT_UNIFORM_TYPE
#else
#define COUNT_UNIFORM_TYPE readonly
#define UNIFORM_TYPE readonly
#endif

layout(set = 2, binding = 0, rg32ui) uniform UNIFORM_TYPE uimage3D
    clusterPointers;
layout(set = 2, binding = 1, r32ui) uniform COUNT_UNIFORM_TYPE uimageBuffer
    lightIndicesCount;
layout(set = 2, binding = 2, r16ui) uniform UNIFORM_TYPE uimageBuffer
    lightIndices;

const float Z_SCALE = 2.4;
const float Z_BIAS = -4.5;

#ifdef WRITE_CULLING_BINDS

float sliceStart(uint slice)
{
    // Inverse of the slice func from Avalanche's Practical Clustered Shading
    return pow(2.0, (slice - Z_BIAS) / Z_SCALE);
}

uvec4 packClusterPointer(uint indexOffset, uint pointCount, uint spotCount)
{
    return uvec4(indexOffset, (pointCount << 16) | spotCount, 0, 0);
}

#else

uvec3 clusterIndex(uvec2 px, float zCam)
{
    // From Avalanche's Practical Clustered Shading
    // Not the exact slices but close
    // -z to match the cam direction
    uint slice =
        min(max(uint(log2(-zCam) * Z_SCALE + Z_BIAS), 0),
            LIGHT_CLUSTER_Z_SLICE_COUNT);
    return uvec3(px / LIGHT_CLUSTER_DIMENSION, slice);
}

void unpackClusterPointer(
    uvec3 index, out uint indexOffset, out uint pointCount, out uint spotCount)
{
    uvec2 packed = imageLoad(clusterPointers, ivec3(index)).xy;
    indexOffset = packed.x;
    pointCount = packed.y >> 16;
    spotCount = packed.y & 0xFFFF;
}

#endif // WRITE_CULLING_BINDS
