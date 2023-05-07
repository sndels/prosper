#ifndef SCENE_VISIBLE_SURFACE_GLSL
#define SCENE_VISIBLE_SURFACE_GLSL

#include "material.glsl"

struct VisibleSurface
{
    vec3 positionWS;
    vec3 normalWS;
    vec3 invViewRayWS;
    float NoV;
    Material material;
};

#endif // SCENE_VISIBLE_SURFACE_GLSL
