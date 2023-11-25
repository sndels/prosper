#ifndef SCENE_VISIBLE_SURFACE_GLSL
#define SCENE_VISIBLE_SURFACE_GLSL

#include "material.glsl"

struct VisibleSurface
{
    vec3 positionWS;
    vec3 normalWS;
    vec3 invViewRayWS;
    vec2 uv;
    float NoV;
    float linearDepth;
    Material material;
};

#endif // SCENE_VISIBLE_SURFACE_GLSL
