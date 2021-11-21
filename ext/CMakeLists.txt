# Pull submodules
execute_process(COMMAND git submodule update --init --depth 1)

# GLM is header only
add_library(glm INTERFACE)
# Define as system to suppress warnings
target_include_directories(glm SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/glm)

# Use ready CMakeLists when available
# Override glfw default options
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target")
add_subdirectory(glfw)

add_library(imgui STATIC "")
target_compile_features(imgui
    PRIVATE
    cxx_std_11
)
target_sources(imgui
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/imgui
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends
    ${Vulkan_INCLUDE_DIR}
)
target_link_libraries(imgui
    PUBLIC
    glfw
    ${Vulkan_LIBRARY}
)

# stb_image is header only
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)

# tinygltf is header only
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)

# vulkan memory allocator is header only
add_library(vma INTERFACE)
# Define as system to suppress warnings
target_include_directories(vma SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/vma/src)

# gli is header only
add_library(gli INTERFACE)
# Define as system to suppress warnings
target_include_directories(gli SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/gli)
