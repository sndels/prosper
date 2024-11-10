#ifndef SHARED_SHADER_STRUCTS_FIELDS_H
#define SHARED_SHADER_STRUCTS_FIELDS_H

#ifdef __cplusplus

#include <glm/glm.hpp>

#define STRUCT_FIELD_SEPARATE_TYPES(glsl_type, cpp_type, name, default_value)  \
    cpp_type name { default_value }

#define STRUCT_FIELD_GLM(type, name, default_value)                            \
    glm::type name { default_value }

#define STRUCT_FIELD(type, name, default_value)                                \
    type name { default_value }

#else // !__cplusplus

#define STRUCT_FIELD_SEPARATE_TYPES(glsl_type, cpp_type, name, default_value)  \
    glsl_type name

#define STRUCT_FIELD_GLM(type, name, default) type name
#define STRUCT_FIELD(type, name, default) type name

#endif // __cplusplus

#endif //  SHARED_SHADER_STRUCTS_FIELDS_H
