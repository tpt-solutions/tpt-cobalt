cmake_minimum_required(VERSION 3.20)
project(tptc VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tptc
  lib/Dialect/TPTIRDialect.cpp
  lib/Dialect/TPTIRTypes.cpp
  lib/Dialect/TPTIROps.cpp
  lib/Parser/TPTAsmParser.cpp
  lib/IR/TPTIRBuilder.cpp
  lib/Pass/TPTIRPasses.cpp
  lib/CodeGen/TPTCodeGen.cpp
  lib/CAPI/tptir_capi.cpp
)

target_include_directories(tptc PUBLIC include)

enable_testing()
add_executable(test_tptir test/TestTPTIR.cpp)
target_link_libraries(test_tptir tptc)
add_test(NAME TPTIR COMMAND test_tptir)
