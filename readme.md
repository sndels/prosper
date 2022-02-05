# prosper
![screenshot](screenshot.png)

Vulkan renderer spun off from following https://vulkan-tutorial.com/. Work of [Sascha Willems](https://github.com/SaschaWillems) also used as reference.

Current features include:
* glTF 2.0 assets
    * Only a very limited subset is currently supported
* Physically based shading
* Tangent space normal mapping
* Mipmapping
* Transparency (no sorting)
* Skybox
* Uncharted 2 -tonemapping

Depends externally on [Vulkan SDK](https://vulkan.lunarg.com/) and the dependencies of [glfw](https://github.com/glfw/glfw). Includes [glm](https://github.com/g-truc/glm), [glfw](https://github.com/glfw/glfw), [libshaderc](https://github.com/google/shaderc), [tinygltf](https://github.com/syoyo/tinygltf) and [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) as submodules. [stb_image](https://github.com/nothings/stb) provided by [tinygltf](https://github.com/syoyo/tinygltf) is also utilized.

Developed mainly on Linux but I check OSX (MoltenVK+AppleClang) and Windows (VS2017) builds periodically.
