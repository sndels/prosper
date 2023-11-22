#ifndef RESTIR_DI_RESERVOIR_GLSL
#define RESTIR_DI_RESERVOIR_GLSL

struct LightReservoir
{
    // <0 signals invalid sample
    int lightIndex;
    float unbiasedContributionWeight;
};

LightReservoir initReservoir()
{
    LightReservoir ret;
    ret.lightIndex = -1;
    ret.unbiasedContributionWeight = 0;

    return ret;
}

LightReservoir unpackReservoir(vec2 packed)
{
    LightReservoir ret;
    ret.lightIndex = floatBitsToInt(packed[0]);
    ret.unbiasedContributionWeight = packed[1];

    return ret;
}

vec2 packReservoir(LightReservoir reservoir)
{
    vec2 packed = vec2(0);
    packed[0] = intBitsToFloat(reservoir.lightIndex);
    packed[1] = reservoir.unbiasedContributionWeight;

    return packed;
}

#endif // RESTIR_DI_RESERVOIR_GLSL
