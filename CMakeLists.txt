cmake_minimum_required(VERSION 3.20)
project(prosper)

option(PROSPER_USE_PCH "Use pre-compiled headers" ON)
option(PROSPER_SETUP_DEPENDENCIES "Pull the in-tree dependencies and correct submodule commits" ON)
option(PROSPER_ALWAYS_O2_DEPENDENCIES "Always build dependencies as optimized" ON)

if(MSVC)
    add_compile_options(/MP)

    if(PROSPER_ALWAYS_O2_DEPENDENCIES)
        # RTC1 (both /RTCs and /RTCu) is incompatible with /O2
        string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
        string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
    endif()
endif() # MSVC

# Set up sub-builds and sources
add_subdirectory(ext)
add_subdirectory(src)

if(PROSPER_ALWAYS_O2_DEPENDENCIES)
    # Only optimize the dependencies that have a noticeable impact
    if(MSVC)
        target_compile_options(meshoptimizer PRIVATE "/O2")
        target_compile_options(mikktspace PRIVATE "/O2")
    else()
        target_compile_options(meshoptimizer PRIVATE "-O2")
        target_compile_options(mikktspace PRIVATE "-O2")
    endif()
endif()

if(LIVEPP_PATH)
    set(LIVEPP_DEFINE "LIVEPP_PATH=L\"${LIVEPP_PATH}\"")
endif()

# Set up project targets
add_executable(prosper ${PROSPER_SOURCES} ${PROSPER_INCLUDES})
target_compile_features(prosper
    PRIVATE
    cxx_std_20
)

if(MSVC)
    # From cppbestpractices
    target_compile_options(prosper
        PRIVATE
        /permissive-
        /Zc:preprocessor
        /W4
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /we4289
        /w14296
        /w14311
        /w14545
        /w14546
        /w14547
        /w14549
        /w14555
        /w14619
        /w14640
        /w14826
        /w14905
        /w14906
        /w14928
        /wd4201 # GLM in PCH bleeds warnings into prosper sources
    )

    if(LIVEPP_PATH)
        target_link_options(prosper PRIVATE
            /FUNCTIONPADMIN
        )
    endif()
else()
    target_compile_options(prosper
        PRIVATE
        -pedantic
        -Wall
        -Wextra
        -Wunused
        -Wno-missing-field-initializers
    )
endif() # NOT MSVC

target_include_directories(prosper
    PRIVATE
    ${PROSPER_INCLUDE_DIR}
    ${LIVEPP_PATH}
)
target_link_libraries(prosper
    PRIVATE
    cxxopts
    glfw
    gli
    glm
    imgui
    ispc_texcomp
    meshoptimizer
    mikktspace
    shaderc
    spirv_headers
    stb_image
    vma
    vulkan
    wheels
)

if(WIN32)
    target_link_libraries(prosper
        PRIVATE
        Dwmapi
    )
endif() # WIN32

target_compile_definitions(prosper
    PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_NO_SETTERS
    VULKAN_HPP_NO_SMART_HANDLE
    VULKAN_HPP_NO_SPACESHIP_OPERATOR
    VULKAN_HPP_NO_STRING
    TINYGLTF_NO_EXTERNAL_IMAGE
    ${LIVEPP_DEFINE}

    # Set up absolute path to resources and binaries
    RES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/"
    BIN_PATH="${CMAKE_CURRENT_BINARY_DIR}/")

if(PROSPER_USE_PCH)
    target_precompile_headers(prosper
        PRIVATE

        [["glm/glm.hpp"]]
        [["glm/gtc/type_ptr.hpp"]]
        [["glm/gtx/component_wise.hpp"]]
        [["glm/gtx/matrix_decompose.hpp"]]
        [["glm/gtx/quaternion.hpp"]]
        [["glm/gtx/transform.hpp"]]
        [["chrono"]]
        [["filesystem"]]
        [["fstream"]]
        [["variant"]]
        [["functional"]]
        [["vulkan/vulkan.hpp"]]
        [["vulkan/vulkan_hash.hpp"]]
        [["wyhash.h"]]
    )
endif() # PROSPER_USE_PCH
