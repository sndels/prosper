#pragma shader_stage(fragment)

#extension GL_ARB_separate_shader_objects : enable

#include "scene/camera.glsl"
#include "scene/skybox.glsl"

layout(location = 0) in vec3 fragTexCoord;
layout(location = 1) in vec4 fragPositionNDC;
layout(location = 2) in vec4 fragPrevPositionNDC;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;

void main()
{
    vec3 color = textureLod(skybox, fragTexCoord, 0).rgb;
    outColor = vec4(color, 1);

    // Store in NDC like in https://alextardif.com/TAA.html
    vec3 posNDC = fragPositionNDC.xyz / fragPositionNDC.w;
    vec3 prevPosNDC = fragPrevPositionNDC.xyz / fragPrevPositionNDC.w;
    vec2 velocity = (posNDC.xy - camera.currentJitter) -
                    (prevPosNDC.xy - camera.previousJitter);
    // Let's have positive motion be upward in the image to try and avoid
    // confusion.
    velocity.y = -velocity.y;

    outVelocity = velocity;
}
