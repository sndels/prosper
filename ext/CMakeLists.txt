if(PROSPER_SETUP_DEPENDENCIES)
    # Pull submodules
    execute_process(COMMAND git submodule update --init --depth 1)
endif()

find_package(Vulkan REQUIRED)
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

add_library(vulkan INTERFACE)
target_include_directories(vulkan SYSTEM INTERFACE ${Vulkan_INCLUDE_DIR})
target_link_libraries(vulkan INTERFACE ${Vulkan_LIBRARY})

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
get_target_property(glfw_include_dirs glfw INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(glfw PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${glfw_include_dirs}")

if(PROSPER_SETUP_DEPENDENCIES)
    message(STATUS "Sync shaderc dependencies")
    find_package(Python COMPONENTS Interpreter)
    execute_process(COMMAND
        ${Python_EXECUTABLE} utils/git-sync-deps
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shaderc
    )
endif()

set(SHADERC_ENABLE_SHARED_CRT ON)
set(SHADERC_SKIP_INSTALL ON)
set(SHADERC_SKIP_TESTS ON)
set(SHADERC_SKIP_EXAMPLES ON)
set(SHADERC_SKIP_COPYRIGHT_CHECK ON)
add_subdirectory(shaderc)
get_target_property(shaderc_include_dirs shaderc INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(shaderc PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${shaderc_include_dirs}")

add_library(spirv_headers INTERFACE)
target_include_directories(spirv_headers SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/shaderc/third_party/spirv-headers/include/spirv/unified1)

add_library(imgui STATIC "")
target_compile_features(imgui
    PRIVATE
    cxx_std_11
)
target_sources(imgui
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui
    SYSTEM
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/imgui
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends
)
target_link_libraries(imgui
    PUBLIC
    glfw
    vulkan
)

# stb_image is header only
add_library(stb_image INTERFACE)
target_include_directories(stb_image SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)

# tinygltf is header only
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tinygltf)

# vulkan memory allocator is header only
add_library(vma INTERFACE)

# Define as system to suppress warnings
target_include_directories(vma SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/vma/include)

# gli is header only
add_library(gli INTERFACE)

# Define as system to suppress warnings
target_include_directories(gli SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/gli)

set(WHEELS_BUILD_TESTS OFF CACHE BOOL "Include tests")
set(WHEELS_BUILD_BENCHES OFF CACHE BOOL "Include benchmarks")
add_subdirectory(wheels)

add_subdirectory(cxxopts)

add_subdirectory(meshoptimizer)

add_library(mikktspace STATIC)
target_sources(mikktspace
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/mikktspace/mikktspace.c
    ${CMAKE_CURRENT_LIST_DIR}/mikktspace/mikktspace.h
)
target_include_directories(imgui
    SYSTEM
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/mikktspace
)

# TODO: Why does compilation fail for any modules that are added after this ispc "include"?
include(ispc_texcomp_CMakeLists.txt)
