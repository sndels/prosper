# Pull submodules
execute_process(COMMAND git submodule update --init --depth 1)

# GLM is header only
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${CMAKE_CURRENT_LIST_DIR}/glm)

# Use ready CMakeLists when available
# Override glfw default options
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target")
add_subdirectory(glfw)

# stb_image is header only
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)

# tinygltf is header only
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)
