// From https://github.com/linebender/piet-gpu/blob/prefix/piet-gpu-hal/examples/shader/prefix.comp
// See https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_memory_scope_semantics : enable
// #extension VK_EXT_shader_atomic_float : require NOT WORKING

#define PI 3.1415926

DEFINE_STRING// This will be (or has been) replaced by constant definitions
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    
layout (local_size_x = THREADS_PER_WORKGROUP, local_size_y = 1, local_size_z = 1 ) in;


void main(){
    
    // subgroupSize is the size of the subgroup – matches the API property
    //gl_SubgroupInvocationID is the ID of the invocation within the subgroup, an integer in the range [0..gl_SubgroupSize).
    // gl_SubgroupID is the ID of the subgroup within the local workgroup, an integer in the range [0..gl_NumSubgroups).
    //gl_NumSubgroups is the number of subgroups within the local workgroup.

    uint workGroup_ix       = gl_WorkGroupID.x;
    uint thread_ix          = gl_LocalInvocationID.x;
    uint workgroupStart_ix  = workGroup_ix*THREADS_PER_WORKGROUP;
    uint absoluteSubgroupId = gl_SubgroupID + gl_NumSubgroups * workGroup_ix;
    uint unique_thread_ix   = absoluteSubgroupId*gl_SubgroupSize + gl_SubgroupInvocationID;
    uint n                  = unique_thread_ix%SIGNAL_LENGTH;
    uint read_ix            = (unique_thread_ix+offset[0])%SIGNAL_LENGTH;
    uint frequency_ix       = unique_thread_ix/SIGNAL_LENGTH;
    
    float Tr = 0;
    float Ti = 0;
    
    float thisF     = f[frequency_ix];
    float thisP     = 1/thisF;
    if(n >= SIGNAL_LENGTH - multiple*thisP){
    //if(n >= SIGNAL_LENGTH - 1024*2){
        float thisDatum = x[read_ix];
        //#if windowed
        //    float w = window[n - uint(SIGNAL_LENGTH - multiple*thisP)];
        //    thisDatum*=w;
        //#endif
        // do the loiacono transform
        float dftlen = 1/sqrt(multiple / thisF);
        
        Tr =  thisDatum*cos(2*PI*thisF*n)*dftlen;
        Ti = -thisDatum*sin(2*PI*thisF*n)*dftlen;
    }
    
    
    // first reduction
    float TrSum = subgroupAdd(Tr);
    float TiSum = subgroupAdd(Ti);
    
    if (subgroupElect()) {
        Lr1[absoluteSubgroupId] = TrSum;
        Li1[absoluteSubgroupId] = TiSum;
    }
    
    // second stage reduction
    if(absoluteSubgroupId >= n*gl_SubgroupSize){
        return;
    }
    
    barrier();
    memoryBarrierBuffer();
    
    TrSum = subgroupAdd(Lr1[unique_thread_ix]);
    TiSum = subgroupAdd(Li1[unique_thread_ix]);
    if (subgroupElect()) {
        Lr0[absoluteSubgroupId] = TrSum;
        Li0[absoluteSubgroupId] = TiSum;
    }

    // third stage reduction
    if(absoluteSubgroupId >= n){
        return;
    }
    
    barrier();
    memoryBarrierBuffer();
    
    TrSum = subgroupAdd(Lr0[unique_thread_ix]);
    TiSum = subgroupAdd(Li0[unique_thread_ix]);
    
    if (subgroupElect()) {
        L[absoluteSubgroupId] = sqrt(TrSum*TrSum + TiSum*TiSum);
        //L[absoluteSubgroupId] = 1;
    }
}