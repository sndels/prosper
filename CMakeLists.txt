cmake_minimum_required(VERSION 3.20)
project(prosper)

if(MSVC)
    add_compile_options(/MP)
endif() # MSVC

# Set up sub-builds and sources
add_subdirectory(ext)
add_subdirectory(include)
add_subdirectory(src)

option(PROSPER_USE_PCH "Use pre-compiled headers" ON)

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
else()
    target_compile_options(prosper
        PRIVATE
        -pedantic
        -Wall
        -Wextra
        -Wunused
    )
endif() # NOT MSVC

target_include_directories(prosper
    PRIVATE
    ${PROSPER_INCLUDE_DIR}
)
target_link_libraries(prosper
    PRIVATE
    glfw
    gli
    glm
    imgui
    shaderc
    stb_image
    vma
    vulkan
)
target_compile_definitions(prosper
    PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_NO_SETTERS
    VULKAN_HPP_NO_SMART_HANDLE
    VULKAN_HPP_NO_SPACESHIP_OPERATOR

    # Set up absolute path to resources and binaries
    RES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/"
    BIN_PATH="${CMAKE_CURRENT_BINARY_DIR}/")

if(PROSPER_USE_PCH)
    target_precompile_headers(prosper
        PRIVATE
        [["chrono"]]
        [["filesystem"]]
        [["fstream"]]
        [["map"]]
        [["set"]]
        [["unordered_map"]]
        [["unordered_set"]]
        [["gli/gli.hpp"]]
        [["glm/glm.hpp"]]
        [["glm/gtc/quaternion.hpp"]]
        [["glm/gtc/matrix_access.hpp"]]
        [["glm/gtc/matrix_transform.hpp"]]
        [["glm/gtc/type_ptr.hpp"]]
        [["glm/gtx/matrix_decompose.hpp"]]
        [["vulkan/vulkan.hpp"]]
        [["vulkan/vulkan_hash.hpp"]]
    )
endif() # PROSPER_USE_PCH
