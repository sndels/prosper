#ifndef COMMON_MATH_GLSL
#define COMMON_MATH_GLSL

#define PI 3.14159265
#define saturate(x) clamp(x, 0.0, 1.0)

#define max2(v) max(v.x, v.y)
#define max3(v) max(max(v.x, v.y), v.z)
#define max4(v) max(max(v.x, v.y), max(v.z, v.w))

#define min2(v) min(v.x, v.y)
#define min3(v) min(min(v.x, v.y), v.z)
#define min4(v) min(min(v.x, v.y), min(v.z, v.w))

#endif // COMMON_MATH_GLSL
