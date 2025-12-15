cmake_minimum_required(VERSION 3.20)
project(mlir-pass-demo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find LLVM / MLIR / CIRCT ---
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

# Include dirs for C++ compile
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${CIRCT_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

# --- TableGen: generate Passes.h.inc ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes.td)

# pass tblgen needs -I for mlir/llvm/circt + local include
set(_TBLGEN_I_ARGS)
foreach(inc_dir ${CIRCT_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  list(APPEND _TBLGEN_I_ARGS -I${inc_dir})
endforeach()

# Generate pass declarations for all passes
mlir_tablegen(Passes.h.inc -gen-pass-decls ${_TBLGEN_I_ARGS})
add_public_tablegen_target(DffDemoPassIncGen)

# --- Build the pass as a module (for potential plugin use) ---
add_library(DffDemoPass MODULE
  lib/Passes.cpp
  lib/QEMUCodeGen.cpp
  lib/ClkAnalysisResult.cpp
)
add_dependencies(DffDemoPass DffDemoPassIncGen)
target_link_libraries(DffDemoPass PRIVATE
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRControlFlowDialect
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSeq
  CIRCTLLHD
)
set_target_properties(DffDemoPass PROPERTIES PREFIX "")

# --- Minimal driver that statically links the pass ---
add_executable(dff-opt
  tool_main.cpp
  lib/Passes.cpp
  lib/QEMUCodeGen.cpp
  lib/ClkAnalysisResult.cpp
)
add_dependencies(dff-opt DffDemoPassIncGen)
target_link_libraries(dff-opt PRIVATE
  # MLIR core
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRAnalysis
  MLIRRewrite
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRControlFlowDialect

  # CIRCT dialects you use
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSeq
  CIRCTLLHD

  # LLVM support (needed by many MLIR libs)
  LLVMSupport
)


target_compile_options(DffDemoPass PRIVATE -fno-rtti)
target_compile_options(dff-opt     PRIVATE -fno-rtti)

