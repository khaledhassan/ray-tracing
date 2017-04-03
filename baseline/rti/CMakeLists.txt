cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_20,code=sm_20
    -gencode arch=compute_37,code=sm_37
#    -O3 --gpu-architecture=sm_37
#        For convenience, in case of simple nvcc compilations, the following shorthand
#        is supported.  If no value for option '--gpu-code' is specified, then the
#        value of this option defaults to the value of '--gpu-architecture'.  In this
#        situation, as only exception to the description above, the value specified
#        for '--gpu-architecture' may be a 'real' architecture (such as a sm_50),
#        in which case nvcc uses the specified 'real' architecture and its closest
#        'virtual' architecture as effective architecture values.  For example, 'nvcc
#        --gpu-architecture=sm_50' is equivalent to 'nvcc --gpu-architecture=compute_50
#        --gpu-code=sm_50,compute_50'.
#        Allowed values for this option:  'compute_20','compute_30','compute_32',
#        'compute_35','compute_37','compute_50','compute_52','compute_53','compute_60',
#        'compute_61','compute_62','sm_20','sm_21','sm_30','sm_32','sm_35','sm_37',
#        'sm_50','sm_52','sm_53','sm_60','sm_61','sm_62'.

    )

# Specify include directories
#include_directories(
#    kernels
#    utility
#    )

# Specify library paths
#link_directories(
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    rti_baseline
    kernel.cu
    gpuRTIStruct.h
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    rti_baseline
    )
