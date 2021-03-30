cmake_minimum_required(VERSION 3.0)

project(simnet_inference CXX C)

#SET(CMAKE_C_COMPILER "/usr/bin/gcc-4.8")
#SET(CMAKE_CXX_COMPILER "/usr/bin/g++-4.8")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(NOT DEFINED PADDLE_LIB)
    message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED INFER_NAME)
    message(FATAL_ERROR "please set INFER_NAME with -DINFER_NAME=INFER_NAME")
endif()

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)

include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")

add_executable(${INFER_NAME} ${INFER_NAME}.cc)

set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.a)

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
    set(DEPS
            ${PADDLE_LIB}/paddle/lib/libpaddle_inference.a)
else()
    set(DEPS
            ${PADDLE_LIB}/paddle/lib/libpaddle_inference.so)
endif()

set(EXTERNAL_LIB "-lrt -ldl -lpthread")

set(DEPS ${DEPS} ${MATH_LIB} glog gflags protobuf z xxhash ${EXTERNAL_LIB})

target_link_libraries(${INFER_NAME} ${DEPS})
