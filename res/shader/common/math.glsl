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

float luminance(vec3 c) { return dot(vec3(0.299, 0.587, 0.114), c); }

// Returns HSV with hue unscaled, multiply it by 60 to get degrees
vec3 rgbToHsv(vec3 rgb)
{
    // https://en.wikipedia.org/wiki/HSL_and_HSV

    float value = max(max(rgb.r, rgb.g), rgb.b);
    float valueMinusChroma = min(min(rgb.r, rgb.g), rgb.b);
    float chroma = value - valueMinusChroma;

    // TODO:
    // Feels like these branches and the value/valueMinusChroma could be folded
    // together
    float hue;
    if (chroma == 0.)
        hue = 0.;
    else if (value == rgb.r)
        hue = mod((rgb.g - rgb.b) / chroma, 6.);
    else if (value == rgb.g)
        hue = (rgb.b - rgb.r) / chroma + 2.;
    else
        hue = (rgb.r - rgb.g) / chroma + 4.;

    float saturation = value == 0. ? 0. : chroma / value;

    return vec3(hue, saturation, value);
}

// Expects HSV with hue not scaled to degrees
vec3 hsvToRgb(vec3 hsv)
{
    // https://en.wikipedia.org/wiki/HSL_and_HSV

    float hue = hsv.r;
    float saturation = hsv.g;
    float value = hsv.b;

    float chroma = value * saturation;

    float x = chroma * (1. - abs(mod(hue, 2.) - 1.));

    // TODO:
    // That's a lot of branching. Is there a clever branchless algo here that's
    // nicer for GPUs?
    vec3 rgb;
    if (hue < 1.)
        rgb = vec3(chroma, x, 0.);
    else if (hue < 2.)
        rgb = vec3(x, chroma, 0.);
    else if (hue < 3.)
        rgb = vec3(0., chroma, x);
    else if (hue < 4.)
        rgb = vec3(0., x, chroma);
    else if (hue < 5.)
        rgb = vec3(x, 0., chroma);
    else
        rgb = vec3(chroma, 0., x);

    float m = value - chroma;

    return rgb + m;
}

uint roundedUpQuotient(uint dividend, uint divisor)
{
    return (dividend - 1) / divisor + 1;
}

#endif // COMMON_MATH_GLSL
