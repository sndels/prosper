layout(push_constant) uniform ScenePC
{
    uint ModelInstanceID;
    uint MeshID;
    uint MaterialID;
    uint DrawType;
}
scenePC;

const uint DrawType_Default = 0;
const uint DrawType_PrimitiveID = 1;
const uint DrawType_MeshID = 2;
const uint DrawType_MaterialID = 3;
