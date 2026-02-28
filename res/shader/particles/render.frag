#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable

#include "../common/dither.glsl"
#include "../shared/shader_structs/push_constants/particles/render.h"

layout(push_constant) uniform PushConstants { RenderPC PC; };

layout(location = 0) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 color = inColor.rgb;
    ivec2 px = ivec2(gl_FragCoord.xy);
    // Cycle through the matrix to get AA for free
    px.x += int(PC.frameIndex) % 8;
    px.y += int(PC.frameIndex) / 8;
    float alpha = ditherAlpha(px, inColor.a);
    if (alpha == 0)
        discard;
    outColor = vec4(color, 1);
}
