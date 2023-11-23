#ifndef RESTIR_DI_RESAMPLING_PHAT_GLSL
#define RESTIR_DI_RESAMPLING_PHAT_GLSL

float pHatLight(VisibleSurface surface, uint lightIndex)
{
    vec3 l;
    float d;
    vec3 irradiance;
    sampleLight(surface, lightIndex, l, d, irradiance);

    vec3 brdf = evalBRDFTimesNoL(l, surface);
    // TODO:
    // Is it ok to skip visiblity here? Sounds reasonable if spatial and
    // temporal reuse don't trace either. Final shading will check visibility on
    // the lucky sample in the reservoir. More noise but no bias compared to
    // tracing visibility for all samples?

    return luminance(irradiance * brdf);
}

#endif // RESTIR_DI_RESAMPLING_PHAT_GLSL
