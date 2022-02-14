cmake_minimum_required(VERSION 3.18)

project(xnnpack_onnx_schema CXX)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)



FetchContent_Declare(
  protobuf_project
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v3.19.4
  SOURCE_SUBDIR cmake
)

# CMake 3.14+
FetchContent_MakeAvailable(protobuf_project)

FetchContent_Declare(
  onnx_project
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.10.2
)

# CMake 3.14+
FetchContent_MakeAvailable(onnx_project)

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

add_library(xnnpack_onnx_schema xnnpack_onnx_defs.cc xnnpack_onnx_defs.h xnnpack_onnx_schema.h xnnpack_opset.h)
onnxruntime_add_include_to_target(xnnpack_onnx_schema onnx onnx_proto protobuf::libprotobuf)
add_dependencies(xnnpack_onnx_schema onnx_proto)