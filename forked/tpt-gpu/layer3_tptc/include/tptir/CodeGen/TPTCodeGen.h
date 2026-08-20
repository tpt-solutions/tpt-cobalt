#ifndef TPTIR_CODEGEN_TPTCODEGEN_H
#define TPTIR_CODEGEN_TPTCODEGEN_H
#include "../Dialect/TPTIRDialect.h"
#include "../Dialect/TPTIROps.h"
#include <sstream>
namespace tptir {
enum class CodeGenTarget : uint8_t { TPTISA, LLVMIR, TPTIRText };
struct CodeGenOptions {
  CodeGenTarget target{CodeGenTarget::TPTISA};
  bool optimize{true}; bool emitComments{true};
  std::string entryFunction{"main"};
};
class TPTCodeGen {
public:
  explicit TPTCodeGen(const CodeGenOptions& opts = CodeGenOptions());
  std::string generate(Region* region);
  std::string generateFromBlocks(const std::vector<Block*>& blocks);
  const std::string& lastError() const { return lastError_; }
private:
  std::string emitTPTISA(Region*);
  std::string emitLLVMIR(Region*);
  std::string emitTPTIRText(Region*);
  std::string emitOperation(const Operation*, std::stringstream&);
  std::string valueName(const Value*) const;
  std::string typeName(const Type*) const;
  CodeGenOptions options_;
  std::string lastError_, stats_;
  uint64_t nextLabelId_{0};
  std::unordered_map<const Value*, std::string> valueNames_;
  std::unordered_map<const Block*, std::string> blockLabels_;
};
}
#endif
