# prosper
![screenshot](screenshot.png)[^1]

Vulkan renderer spun off from following https://vulkan-tutorial.com/. Work of [Sascha Willems](https://github.com/SaschaWillems) and [Ray Tracing Gems II](https://developer.nvidia.com/ray-tracing-gems-ii) also used as reference.

Current features include:
* glTF 2.0 assets
    * Only a very limited subset is currently supported
* Physically based shading
* Tangent space normal mapping
* Mipmapping
* Transparency (no sorting)
* Skybox
* 1D ACES-tonemap[^2]
* Clustered lighting
  * Points, spots
  * Sphere bounds
  * View space clusters with depth slices
* Bindless materials
* Bindless geometry
* Pipeline ray tracing
  * ID debug only for now
* Polling shader recompilation
* Scoped profiling
  * GPU with timestamps
  * CPU with `std::chrono`
  * Should be 1:1 mapping between the GPU frame and the CPU frame that recorded it

Depends externally on [Vulkan SDK](https://vulkan.lunarg.com/) and the dependencies of [glfw](https://github.com/glfw/glfw). Includes [glfw](https://github.com/glfw/glfw), [gli](https://github.com/g-truc/gli), [glm](https://github.com/g-truc/glm), [imgui](https://github.com/ocornut/imgui), [libshaderc](https://github.com/google/shaderc), [tinygltf](https://github.com/syoyo/tinygltf) and [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) as submodules.

[^1]:Scene modified from Crytek Sponza, originally by Frank Meinl with tweaks by Morgan McGuire, Alexandre Pestana and the authors of [glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/Sponza)
[^2]:From a blog post by [Krzysztof Narkowicz](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve)
