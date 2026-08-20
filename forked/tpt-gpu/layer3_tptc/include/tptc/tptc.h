#ifndef TPTC_TPTC_H
#define TPTC_TPTC_H
#include "../tptir/Dialect/TPTIRDialect.h"
#include "../tptir/Dialect/TPTIRTypes.h"
#include "../tptir/Dialect/TPTIROps.h"
#include "../tptir/Pass/TPTIRPasses.h"
#include "../tptir/CodeGen/TPTCodeGen.h"
namespace tptc {
struct CompilerConfig {
  std::string entryFunction{"main"};
  CodeGenTarget target{CodeGenTarget::TPTISA};
  bool optimize{true};
};
struct CompileResult {
  bool success{false}; std::string output;
  std::string errors; size_t numInstructions{0};
};
CompileResult compile(const std::string& src, const CompilerConfig& cfg = CompilerConfig{});
bool compileFile(const std::string& inp, const std::string& out, const CompilerConfig& cfg = CompilerConfig{});
std::string version();
}
#endif
