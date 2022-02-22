cmake_minimum_required(VERSION 3.18)

project(xnnpack_onnx_schema CXX)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

option(onnxruntime_USE_KLEE "Build with KLEE fuzzer" OFF)

set(Protobuf_USE_STATIC_LIBS ON)
set(ONNX_USE_LITE_PROTO ON CACHE BOOL "" FORCE)
set(ONNX_DISABLE_EXCEPTIONS  ON CACHE BOOL "" FORCE)
set(ONNX_DISABLE_STATIC_REGISTRATION  ON CACHE BOOL "" FORCE)
set(XNNPACK_ONNX_PROTOBUF_URL https://github.com/protocolbuffers/protobuf.git)
set(XNNPACK_ONNX_PROTOBUF_VERSION v3.19.4)
set(XNNPACK_ONNX_ONNX_URL https://github.com/onnx/onnx.git)
set(XNNPACK_ONNX_ONNX_VERSION 9b3524511003e11998d5b58ec9d0add3ce568506)

if(onnxruntime_USE_KLEE)
  set(CMAKE_CXX_LINK_EXECUTABLE "llvm-link <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  # Disable C++ exceptions.
  string(REGEX REPLACE "-fexceptions" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
  add_definitions(-DONNX_NO_EXCEPTIONS=1 -D_GLIBCXX_DEBUG)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>")
   #-g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__ -I /opt/klee/klee_build120stp_z3/include -emit-llvm>")
  add_compile_options(-g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__ -I /opt/klee/klee_build120stp_z3/include -emit-llvm)
  set(XNNPACK_ONNX_PROTOBUF_URL https://github.com/snnn/protobuf.git)
  set(XNNPACK_ONNX_PROTOBUF_VERSION 61718cd6748912b8aa1e7e665f5538182b7e494e)
endif()

FetchContent_Declare(
  protobuf_project
  GIT_REPOSITORY ${XNNPACK_ONNX_PROTOBUF_URL}
  GIT_TAG        ${XNNPACK_ONNX_PROTOBUF_VERSION}
  SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(protobuf_project)
FetchContent_Declare(
  onnx_project
  GIT_REPOSITORY ${XNNPACK_ONNX_ONNX_URL}
  GIT_TAG        ${XNNPACK_ONNX_ONNX_VERSION}
)
FetchContent_MakeAvailable(onnx_project)

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

add_library(xnnpack_onnx_schema xnnpack_onnx_defs.cc xnnpack_onnx_defs.h xnnpack_onnx_schema.h xnnpack_opset.h)
onnxruntime_add_include_to_target(xnnpack_onnx_schema onnx onnx_proto protobuf::libprotobuf-lite)
add_dependencies(xnnpack_onnx_schema onnx_proto)

if(onnxruntime_USE_KLEE)
  #add_executable(test1 tests/main.cc $<TARGET_OBJECTS:xnnpack_onnx_schema> $<TARGET_OBJECTS:onnx> $<TARGET_OBJECTS:onnx_proto> $<TARGET_OBJECTS:protobuf::libprotobuf-lite>)
  add_executable(test1 tests/main.cc $<TARGET_OBJECTS:onnx> $<TARGET_OBJECTS:onnx_proto> $<TARGET_OBJECTS:protobuf::libprotobuf-lite>)  
  onnxruntime_add_include_to_target(test1 xnnpack_onnx_schema onnx onnx_proto protobuf::libprotobuf-lite)
  #target_link_libraries(test1 xnnpack_onnx_schema onnx onnx_proto protobuf::libprotobuf)
  target_include_directories(test1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
