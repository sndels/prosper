#ifndef PARTICLES_FREELIST_GLSL
#define PARTICLES_FREELIST_GLSL

int freelistPushIndex()
{
    uvec4 activeThreadsMask = subgroupBallot(true);
    int activeThreadCount = int(subgroupBallotBitCount(activeThreadsMask));
    int subgroupStartOffset;
    if (subgroupElect())
    {
        subgroupStartOffset =
            // Freelist overflow here likely means a double free happened
            // somewhere
            atomicAdd(inOutParticlesFreelist.count, activeThreadCount);
    }
    subgroupStartOffset = subgroupBroadcastFirst(subgroupStartOffset);

    // offset within the subgroup block
    return subgroupStartOffset +
           int(subgroupBallotExclusiveBitCount(activeThreadsMask));
}

int freelistPopIndex()
{
    uvec4 activeThreadsMask = subgroupBallot(true);
    int activeThreadCount = int(subgroupBallotBitCount(activeThreadsMask));
    int subgroupStartOffset;
    if (subgroupElect())
    {
        subgroupStartOffset =
            atomicAdd(inOutParticlesFreelist.count, -activeThreadCount);
        if (subgroupStartOffset < activeThreadCount)
        {
            subgroupStartOffset = -activeThreadCount - 1;
            atomicExchange(inOutParticlesFreelist.count, 0);
        }
    }
    subgroupStartOffset = subgroupBroadcastFirst(subgroupStartOffset);

    // Offset within the subgroup block
    return subgroupStartOffset -
           int(subgroupBallotExclusiveBitCount(activeThreadsMask));
}

#endif // PARTICLES_FREELIST_GLSL
