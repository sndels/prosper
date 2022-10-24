#ifndef GEOMETRY_GLSL
#define GEOMETRY_GLSL

struct MeshBuffer
{
    uint index;
    uint offset;
};

struct MeshBuffers
{
    MeshBuffer indices;
    MeshBuffer positions;
    MeshBuffer normals;
    MeshBuffer tangents;
    MeshBuffer texCoord0s;
    uint usesShortIndices;
};
layout(std430, set = GEOMETRY_SET, binding = 0) readonly buffer
    MeshBuffersBuffer
{
    MeshBuffers data[];
}
meshBuffersBuffer;
// Unbounded array requires GL_EXT_nonuniform_qualifier even if it is indexed
// with a uniform index
layout(std430, set = GEOMETRY_SET, binding = 1) readonly buffer GeometryBuffers
{
    uint data[];
}
geometryBuffers[];

uint loadIndex(MeshBuffer b, uint index, uint usesShortIndices)
{
    if (usesShortIndices == 1)
    {
        uint i = geometryBuffers[b.index].data[b.offset + (index / 2)];
        return (i >> ((index & 1) * 16)) & 0xFFFF;
    }
    else
        return geometryBuffers[b.index].data[b.offset + index];
}

float loadFloat(MeshBuffer b, uint index)
{
    return uintBitsToFloat(geometryBuffers[b.index].data[b.offset + index]);
}

vec2 loadVec2(MeshBuffer b, uint index)
{
    return b.index < 0xFFFFFFF
               ? vec2(loadFloat(b, 2 * index), loadFloat(b, 2 * index + 1))
               : vec2(0);
}

vec3 loadVec3(MeshBuffer b, uint index)
{
    return b.index < 0xFFFFFFF
               ? vec3(
                     loadFloat(b, 3 * index), loadFloat(b, 3 * index + 1),
                     loadFloat(b, 3 * index + 2))
               : vec3(0);
}

vec4 loadVec4(MeshBuffer b, uint index)
{
    return b.index < 0xFFFFFFF
               ? vec4(
                     loadFloat(b, 4 * index), loadFloat(b, 4 * index + 1),
                     loadFloat(b, 4 * index + 2), loadFloat(b, 4 * index + 3))
               : vec4(0);
}

struct Vertex
{
    vec3 Position;
    vec3 Normal;
    vec4 Tangent;
    vec2 TexCoord0;
};

Vertex loadVertex(uint meshID, uint index)
{
    MeshBuffers buffers = meshBuffersBuffer.data[meshID];

    uint vertexIndex =
        loadIndex(buffers.indices, index, buffers.usesShortIndices);

    Vertex ret;

    ret.Position = loadVec3(buffers.positions, vertexIndex);
    ret.Normal = loadVec3(buffers.normals, vertexIndex);
    ret.Tangent = loadVec4(buffers.tangents, vertexIndex);
    ret.TexCoord0 = loadVec2(buffers.texCoord0s, vertexIndex);

    return ret;
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

#endif // GEOMETRY_GLSL
