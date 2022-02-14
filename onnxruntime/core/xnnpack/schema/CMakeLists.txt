cmake_minimum_required(VERSION 3.18)

project(xnnpack_onnx_schema CXX)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

option(onnxruntime_USE_KLEE "Build with KLEE fuzzer" ON)

set(Protobuf_USE_STATIC_LIBS ON)
set(ONNX_USE_LITE_PROTO ON CACHE BOOL "" FORCE)
set(ONNX_DISABLE_EXCEPTIONS  ON CACHE BOOL "" FORCE)

if(onnxruntime_USE_KLEE)

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>")
   #-g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__ -I /opt/klee/klee_build120stp_z3/include -emit-llvm>")
  add_compile_options(-g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__ -I /opt/klee/klee_build120stp_z3/include -emit-llvm)
  #string(APPEND CMAKE_CXX_FLAGS "-stdlib=libc++ -g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__ -I /opt/klee/klee_build120stp_z3/include -emit-llvm")
  #string(APPEND CMAKE_C_FLAGS "-g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__ -I /opt/klee/klee_build120stp_z3/include -emit-llvm")
    FetchContent_Declare(
    protobuf_project
    GIT_REPOSITORY https://github.com/snnn/protobuf.git
    GIT_TAG        61718cd6748912b8aa1e7e665f5538182b7e494e
    SOURCE_SUBDIR cmake
  )

  FetchContent_MakeAvailable(protobuf_project)

endif()

FetchContent_Declare(
  protobuf_project
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v3.19.4
  SOURCE_SUBDIR cmake
)

FetchContent_MakeAvailable(protobuf_project)

FetchContent_Declare(
  onnx_project
  GIT_REPOSITORY https://github.com/snnn/onnx.git
  GIT_TAG        b1d2b2fd8557604a6b387f5b6579fed9dc7991a4
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

add_executable(test1 tests/main.cc)
target_link_libraries(test1 onnx onnx_proto protobuf::libprotobuf)
