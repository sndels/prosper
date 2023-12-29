#ifndef PROSPER_SCENE_FWD_HPP
#define PROSPER_SCENE_FWD_HPP

// Accessors.hpp
class TimeAccessor;
struct KeyFrameInterpolation;

// Animations.hpp
struct Animations;

// Camera.hpp
class Camera;
struct CameraOffset;
struct CameraParameters;
struct CameraTransform;
struct CameraUniforms;
struct PerspectiveParameters;

// DebugGeometry.hpp
struct DebugLines;

// DeferredLoadingContext.hpp
struct DeferredLoadingContext;

// Lights.hpp
struct DirectionalLight;
struct PointLight;
struct PointLights;
struct SpotLight;
struct SpotLights;

// Material.hpp
struct Material;
struct Texture2DSampler;

// Mesh.hpp
struct MeshBuffers;
struct MeshInfo;

// Model.hpp
struct Model;
struct ModelInstance;

// Scene.hpp
struct Scene;

// Texture.hpp
class Texture;
class Texture2D;
class TextureCubemap;

// World.hpp
class World;

// WorldData.hpp
class WorldData;

// WorldRenderStructs.hpp
struct WorldDSLayouts;
struct WorldByteOffsets;
struct WorldDescriptorSets;
struct SkyboxResources;

#endif // PROSPER_SCENE_FWD_HPP
