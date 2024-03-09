#ifndef DOF_BILATERAL_GLSL
#define DOF_BILATERAL_GLSL

float bilateralWeight(float outputCoC, float sampleCoC)
{
    // return saturate(1 - abs(outputCoC - sampleCoC));
    // This is from Abadie, seems to help make thin geometry breakage have it
    // disappear instead of making the full weight 0 causing NaNs (e.g. light
    // debug frames)
    return saturate(1 - (outputCoC - sampleCoC));
}

struct BilateralInput
{
    vec4 illuminationWeight01;
    vec4 illuminationWeight11;
    vec4 illuminationWeight10;
    vec4 illuminationWeight00;
};

vec4 bilateralFilter(BilateralInput inputs)
{
    // TODO: Min/Max instead of average?
    float cocOut =
        (inputs.illuminationWeight01.a + inputs.illuminationWeight11.a +
         inputs.illuminationWeight10.a + inputs.illuminationWeight00.a) /
        4;
    cocOut =
        min(min(inputs.illuminationWeight01.a, inputs.illuminationWeight11.a),
            min(inputs.illuminationWeight10.a, inputs.illuminationWeight00.a));

    // TODO: Is this how you bilateral filter?

    float bilateralWeight01 =
        bilateralWeight(cocOut, inputs.illuminationWeight01.a);
    float bilateralWeight11 =
        bilateralWeight(cocOut, inputs.illuminationWeight11.a);
    float bilateralWeight10 =
        bilateralWeight(cocOut, inputs.illuminationWeight10.a);
    float bilateralWeight00 =
        bilateralWeight(cocOut, inputs.illuminationWeight00.a);
    float bilateralNormalization = bilateralWeight01 + bilateralWeight11 +
                                   bilateralWeight10 + bilateralWeight00;

    vec4 weighed01 = bilateralWeight01 * inputs.illuminationWeight01;
    vec4 weighed11 = bilateralWeight11 * inputs.illuminationWeight11;
    vec4 weighed10 = bilateralWeight10 * inputs.illuminationWeight10;
    vec4 weighed00 = bilateralWeight00 * inputs.illuminationWeight00;

    return (weighed01 + weighed11 + weighed10 + weighed00) /
           bilateralNormalization;
}

#endif // DOF_BILATERAL_GLSL
