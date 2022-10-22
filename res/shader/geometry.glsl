#ifndef GEOMETRY_GLSL
#define GEOMETRY_GLSL

// Unbounded array requires GL_EXT_nonuniform_qualifier even if it is indexed
// with a uniform index
layout(std430, set = VERTEX_BUFFERS_SET, binding = 0) readonly buffer
    VertexBuffers
{
    float data[];
}
vertexBuffers[];

// Unbounded array requires GL_EXT_nonuniform_qualifier even if it is indexed
// with a uniform index
layout(std430, set = INDEX_BUFFERS_SET, binding = 0) readonly buffer
    IndexBuffers
{
    uint data[];
}
indexBuffers[];

struct Vertex
{
    vec3 Position;
    vec3 Normal;
    vec4 Tangent;
    vec2 TexCoord0;
};
#define VERTEX_FLOATS (3 + 3 + 4 + 2)
#define VERTEX_POS_FLOAT_OFFSET 0
#define VERTEX_NORMAL_FLOAT_OFFSET 3
#define VERTEX_TANGENT_FLOAT_OFFSET (3 + 3)
#define VERTEX_TEXCOORD0_FLOAT_OFFSET (3 + 3 + 4)

Vertex loadVertex(uint meshID, uint index)
{
    uint vertexIndex = indexBuffers[meshID].data[index];
    uint vertexOffset = vertexIndex * VERTEX_FLOATS;

    Vertex ret;

    ret.Position = vec3(
        vertexBuffers[meshID].data[vertexOffset + VERTEX_POS_FLOAT_OFFSET + 0],
        vertexBuffers[meshID].data[vertexOffset + VERTEX_POS_FLOAT_OFFSET + 1],
        vertexBuffers[meshID].data[vertexOffset + VERTEX_POS_FLOAT_OFFSET + 2]);
    ret.Normal = vec3(
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_NORMAL_FLOAT_OFFSET + 0],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_NORMAL_FLOAT_OFFSET + 1],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_NORMAL_FLOAT_OFFSET + 2]);
    ret.Tangent = vec4(
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 0],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 1],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 2],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TANGENT_FLOAT_OFFSET + 3]);
    ret.TexCoord0 = vec2(
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TEXCOORD0_FLOAT_OFFSET + 0],
        vertexBuffers[meshID]
            .data[vertexOffset + VERTEX_TEXCOORD0_FLOAT_OFFSET + 1]);

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
