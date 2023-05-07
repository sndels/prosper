#ifndef SCENE_LIGHT_CLUSTERS_GLSL
#define SCENE_LIGHT_CLUSTERS_GLSL

#ifdef WRITE_CULLING_BINDS
#define UNIFORM_TYPE writeonly
#define COUNT_UNIFORM_TYPE
#else
#define COUNT_UNIFORM_TYPE readonly
#define UNIFORM_TYPE readonly
#endif

layout(set = LIGHT_CLUSTERS_SET, binding = 0, rg32ui) uniform
    UNIFORM_TYPE uimage3D clusterPointers;
layout(set = LIGHT_CLUSTERS_SET, binding = 1, r32ui) uniform COUNT_UNIFORM_TYPE
    uimageBuffer lightIndicesCount;
layout(set = LIGHT_CLUSTERS_SET, binding = 2, r16ui) uniform UNIFORM_TYPE
    uimageBuffer lightIndices;

const float Z_SCALE = 2.4;
const float Z_BIAS = -4.5;

#ifdef WRITE_CULLING_BINDS

float sliceStart(uint slice)
{
    // Inverse of Doom 2016 depth slice func
    // https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
    float sliceFrac = float(slice) / float(LIGHT_CLUSTER_Z_SLICE_COUNT);
    return camera.near * pow(camera.far / camera.near, sliceFrac);
}

uvec4 packClusterPointer(uint indexOffset, uint pointCount, uint spotCount)
{
    return uvec4(indexOffset, (pointCount << 16) | spotCount, 0, 0);
}

#else

uvec3 clusterIndex(uvec2 px, float zCam)
{
    // Doom 2016 depth slice func
    // https://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf
    // -z to match the cam direction
    uint slice =
        max(uint(
                LIGHT_CLUSTER_Z_SLICE_COUNT * log(-zCam / camera.near) /
                log(camera.far / camera.near)),
            0);
    return uvec3(px / LIGHT_CLUSTER_DIMENSION, slice);
}

struct LightClusterInfo
{
    uint indexOffset;
    uint pointCount;
    uint spotCount;
};

LightClusterInfo unpackClusterPointer(uvec2 px, float zCam)
{
    uvec3 index = clusterIndex(px, zCam);
    uvec2 packed = imageLoad(clusterPointers, ivec3(index)).xy;

    LightClusterInfo ret;
    ret.indexOffset = packed.x;
    ret.pointCount = packed.y >> 16;
    ret.spotCount = packed.y & 0xFFFF;

    return ret;
}

#endif // WRITE_CULLING_BINDS

#endif // SCENE_LIGHT_CLUSTERS_GLSL
