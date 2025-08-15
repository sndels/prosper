#ifndef SHARED_SHADER_STRUCTS_SCENE_FWD_HPP
#define SHARED_SHADER_STRUCTS_SCENE_FWD_HPP

#ifndef __cplusplus
#error "Forward declarations have no meaning in GLSL"
#endif // !__cplusplus

namespace scene::shader_structs
{

struct CameraUniforms;
struct DrawInstance;
struct GeometryMetadata;
struct DirectionalLightParameters;
struct PointLight;
struct SpotLight;
struct Texture2DSampler;
struct MaterialData;
struct ModelInstanceTransforms;

} // namespace scene::shader_structs

#endif // SHARED_SHADER_STRUCTS_SCENE_FWD_HPP
