#ifndef VISIBLE_SURFACE_GLSL
#define VISIBLE_SURFACE_GLSL

#include "material.glsl"

struct VisibleSurface
{
    vec3 positionWS;
    vec3 normalWS;
    vec3 invViewRayWS;
    float NoV;
    Material material;
};

#endif // VISIBLE_SURFACE_GLSL
