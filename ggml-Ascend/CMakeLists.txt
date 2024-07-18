# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(ACLNN_EXAMPLE)

# Compile options
add_compile_options(-std=c++11)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(BASE_LLAMA_DIR "/home/powerinfer/Desktop/tz/llama/llama.cpp")
file(GLOB SRC_LIST ${BASE_LLAMA_DIR}/*.cpp ${BASE_LLAMA_DIR}/*.h ${BASE_LLAMA_DIR}/*.c)

# 设置可执行文件名（如test_1），并指定待运行算子文件*.cpp所在目录
add_executable(test
        test.cpp
        aclnn-test.cpp
        aclnn-compute.cpp
        aclnn-compute-test.cpp
        aclnn-add.cpp
        aclnn-add-test.cpp
        aclnn-unary.cpp
        aclnn-unary-test.cpp
        aclnn-mul.cpp
        aclnn-mul-test.cpp
        aclnn-cpy.cpp
        aclnn-cpy-test.cpp
        aclnn-comp.cpp
        aclnn-unary.cpp
        aclnn-norm.cpp
        aclnn-comp.cpp
        aclnn-math.cpp
        aclnn-rope.cpp
        aclnn-permute
        common.cpp)

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/home/powerinfer/Ascend/ascend-toolkit/latest")
endif()
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
        ..
)

# 设置链接的库文件路径
target_link_libraries(test PRIVATE
        ${ASCEND_PATH}/lib64/libascendcl.so
        ${ASCEND_PATH}/lib64/libnnopbase.so
        ${ASCEND_PATH}/lib64/libopapi.so)

# 可执行文件在CMakeLists文件所在目录的bin目录下
install(TARGETS test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})