# Have this here instead of subdirectories to make wildcard clang-format simpler before the
# llvm version is new enough for .clang-format-ignores
set(SHARED_SHADER_STRUCTS_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/bloom/separate.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/dof/dilate.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/dof/gather.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/dof/reduce.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/dof/setup.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/restir_di/initial_reservoirs.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/restir_di/spatial_reuse.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/restir_di/trace.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/deferred_shading.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/draw_list_culler.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/draw_list_generator.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/forward.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/gbuffer.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/hiz_downsampler.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/light_clustering.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/prefilter_radiance.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/rt_reference.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/taa_resolve.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/texture_debug.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/texture_readback.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/push_constants/tone_map.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/camera.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/draw_instance.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/geometry_metadata.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/lights.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/material_data.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/scene/model_instance_transforms.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/fields.h
    ${CMAKE_CURRENT_LIST_DIR}/shader/shared/shader_structs/fwd.hpp
    PARENT_SCOPE
)

add_library(shared_shader_structs INTERFACE ${SHARED_SHADER_STRUCTS_INCLUDES})
target_include_directories(shared_shader_structs SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/shader/shared)
target_link_libraries(shared_shader_structs INTERFACE glm wheels)
