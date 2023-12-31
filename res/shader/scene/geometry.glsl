#ifndef SCENE_GEOMETRY_GLSL
#define SCENE_GEOMETRY_GLSL

#include "vertex.glsl"

struct MeshBuffer
{
    uint index;
    uint offset;
};

struct MeshBuffers
{
    uint bufferIndex;
    // All of these offsets are into the data interpreted as a u32 'array'
    // offset of 0xFFFFFFFF signals an unused attribute
    uint indicesOffset;
    uint positionsOffset;
    uint normalsOffset;
    uint tangentsOffset;
    uint texCoord0sOffset;
    uint usesShortIndices;
};
layout(std430, set = GEOMETRY_SET, binding = 0) readonly buffer
    MeshBuffersBuffer
{
    MeshBuffers data[];
}
meshBuffersBuffer;

layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer GeometryBuffers
{
    uint data[];
}
geometryBuffers[];

#ifdef NON_UNIFORM_GEOMETRY_BUFFER_INDICES
#define GET_GEOMETRY_BUFFER(index) geometryBuffers[nonuniformEXT(index)]
#else // !NON_UNIFORM_GEOMETRY_BUFFER_INDICES
#define GET_GEOMETRY_BUFFER(index) geometryBuffers[index]
#endif // NON_UNIFORM_GEOMETRY_BUFFER_INDICES

uint loadIndex(
    uint bufferIndex, uint bufferOffset, uint index, uint usesShortIndices)
{
    if (usesShortIndices == 1)
    {
        uint i =
            GET_GEOMETRY_BUFFER(bufferIndex).data[bufferOffset + (index / 2)];
        return (i >> ((index & 1) * 16)) & 0xFFFF;
    }
    else
        return GET_GEOMETRY_BUFFER(bufferIndex).data[bufferOffset + index];
}

float loadFloat(uint bufferIndex, uint bufferOffset, uint index)
{
    return uintBitsToFloat(
        GET_GEOMETRY_BUFFER(bufferIndex).data[bufferOffset + index]);
}

vec2 loadVec2(uint bufferIndex, uint bufferOffset, uint index)
{
    return bufferOffset < 0xFFFFFFFF
               ? vec2(
                     loadFloat(bufferIndex, bufferOffset, 2 * index),
                     loadFloat(bufferIndex, bufferOffset, 2 * index + 1))
               : vec2(0);
}

vec3 loadVec3(uint bufferIndex, uint bufferOffset, uint index)
{
    return bufferOffset < 0xFFFFFFFF
               ? vec3(
                     loadFloat(bufferIndex, bufferOffset, 3 * index),
                     loadFloat(bufferIndex, bufferOffset, 3 * index + 1),
                     loadFloat(bufferIndex, bufferOffset, 3 * index + 2))
               : vec3(0);
}

vec4 loadVec4(uint bufferIndex, uint bufferOffset, uint index)
{
    return bufferOffset < 0xFFFFFFFF
               ? vec4(
                     loadFloat(bufferIndex, bufferOffset, 4 * index),
                     loadFloat(bufferIndex, bufferOffset, 4 * index + 1),
                     loadFloat(bufferIndex, bufferOffset, 4 * index + 2),
                     loadFloat(bufferIndex, bufferOffset, 4 * index + 3))
               : vec4(0);
}

Vertex loadVertex(uint meshID, uint index)
{
    MeshBuffers buffers = meshBuffersBuffer.data[meshID];

    uint vertexIndex = loadIndex(
        buffers.bufferIndex, buffers.indicesOffset, index,
        buffers.usesShortIndices);

    Vertex ret;

    ret.Position =
        loadVec3(buffers.bufferIndex, buffers.positionsOffset, vertexIndex);
    ret.Normal =
        loadVec3(buffers.bufferIndex, buffers.normalsOffset, vertexIndex);
    ret.Tangent =
        loadVec4(buffers.bufferIndex, buffers.tangentsOffset, vertexIndex);
    ret.TexCoord0 =
        loadVec2(buffers.bufferIndex, buffers.texCoord0sOffset, vertexIndex);

    return ret;
}

vec2 loadUV(uint meshID, uint index)
{
    MeshBuffers buffers = meshBuffersBuffer.data[meshID];

    uint vertexIndex = loadIndex(
        buffers.bufferIndex, buffers.indicesOffset, index,
        buffers.usesShortIndices);

    return loadVec2(buffers.bufferIndex, buffers.texCoord0sOffset, vertexIndex);
}

vec2 baryInterpolate(vec2 v0, vec2 v1, vec2 v2, float a, float b, float c)
{
    return v0 * a + v1 * b + v2 * c;
}
vec3 baryInterpolate(vec3 v0, vec3 v1, vec3 v2, float a, float b, float c)
{
    return v0 * a + v1 * b + v2 * c;
}
vec4 baryInterpolate(vec4 v0, vec4 v1, vec4 v2, float a, float b, float c)
{
    return v0 * a + v1 * b + v2 * c;
}

Vertex interpolate(Vertex v0, Vertex v1, Vertex v2, vec2 baryCoord)
{
    float a = 1 - baryCoord.x - baryCoord.y;
    float b = baryCoord.x;
    float c = baryCoord.y;

    Vertex ret;
    ret.Position =
        baryInterpolate(v0.Position, v1.Position, v2.Position, a, b, c);
    ret.Normal = baryInterpolate(v0.Normal, v1.Normal, v2.Normal, a, b, c);
    ret.Tangent = baryInterpolate(v0.Tangent, v1.Tangent, v2.Tangent, a, b, c);
    ret.TexCoord0 =
        baryInterpolate(v0.TexCoord0, v1.TexCoord0, v2.TexCoord0, a, b, c);

    return ret;
}

vec2 interpolate(vec2 v0, vec2 v1, vec2 v2, vec2 baryCoord)
{
    float a = 1 - baryCoord.x - baryCoord.y;
    float b = baryCoord.x;
    float c = baryCoord.y;

    return baryInterpolate(v0, v1, v2, a, b, c);
}

#endif // SCENE_GEOMETRY_GLSL
