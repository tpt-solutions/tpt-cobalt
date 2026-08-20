// =============================================================================
// TPTIROps.h — TPTIR Operation Definitions
// =============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// =============================================================================

#ifndef TPTIR_DIALECT_TPTIROPS_H
#define TPTIR_DIALECT_TPTIROPS_H

#include "TPTIRDialect.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace tptir {

enum class OpCategory : uint8_t {
  Arithmetic, Logical, Comparison, Memory, Tensor,
  VectorSIMT, ControlFlow, Conversion, Special, Constant, Other
};

struct OpId {
  std::string name;
  OpCategory category;
  OpId() : category(OpCategory::Other) {}
  OpId(const std::string& n, OpCategory cat = OpCategory::Other)
    : name(n), category(cat) {}
  bool operator==(const OpId& other) const { return name == other.name; }
};

class Value {
public:
  Value() : type_(nullptr), id_(0) {}
  Value(Type* type, uint64_t id) : type_(type), id_(id) {}
  Type* type() const { return type_; }
  uint64_t id() const { return id_; }
  bool isValid() const { return type_ != nullptr; }
  std::string toString() const;
private:
  Type* type_;
  uint64_t id_;
};

class Operation {
public:
  Operation(OpId opId, std::vector<Value*> operands, Type* resultType = nullptr);
  virtual ~Operation();
  const OpId& opId() const { return opId_; }
  const std::string& name() const { return opId_.name; }
  OpCategory category() const { return opId_.category; }
  const std::vector<Value*>& operands() const { return operands_; }
  Value* getOperand(size_t i) const { return operands_[i]; }
  size_t numOperands() const { return operands_.size(); }
  Type* resultType() const { return resultType_; }
  void setResultType(Type* t) { resultType_ = t; }
  Value* result() const { return result_; }
  void setResult(Value* v) { result_ = v; }
  void setAttr(const std::string& key, const std::string& value);
  std::string getAttr(const std::string& key, const std::string& def = "") const;
  bool hasAttr(const std::string& key) const;
  virtual std::string toString() const;
  virtual Operation* clone() const;
private:
  OpId opId_;
  std::vector<Value*> operands_;
  Type* resultType_;
  Value* result_;
  std::unordered_map<std::string, std::string> attrs_;
};

class Block {
public:
  Block(const std::string& label = "");
  ~Block();
  const std::string& label() const { return label_; }
  void setLabel(const std::string& l) { label_ = l; }
  void addOperation(Operation* op);
  Operation* getOperation(size_t i) const;
  size_t numOperations() const { return ops_.size(); }
  const std::vector<Operation*>& operations() const { return ops_; }
  Operation* getTerminator() const;
  void addArgument(Value* arg);
  const std::vector<Value*>& arguments() const { return args_; }
  size_t numArguments() const { return args_.size(); }
  std::string toString() const;
private:
  std::string label_;
  std::vector<Operation*> ops_;
  std::vector<Value*> args_;
};

class Region {
public:
  Region() = default;
  ~Region();
  void addBlock(Block* block);
  Block* getBlock(size_t i) const;
  size_t numBlocks() const { return blocks_.size(); }
  const std::vector<Block*>& blocks() const { return blocks_; }
  std::string toString() const;
private:
  std::vector<Block*> blocks_;
};

Operation* createArithOp(const std::string& opName, Value* lhs, Value* rhs, Type* resultType);
Operation* createConstantOp(Type* type, const std::string& valueStr);
Operation* createLoadOp(Value* memref, Type* resultType);
Operation* createStoreOp(Value* value, Value* memref);
Operation* createMMAOp(Value* a, Value* b, Value* c, Type* resultType);
Operation* createVectorOp(const std::string& opName, Value* lhs, Value* rhs, Type* resultType);
Operation* createBranchOp(Block* target);
Operation* createCondBranchOp(Value* condition, Block* trueBlock, Block* falseBlock);
Operation* createReturnOp(const std::vector<Value*>& values = {});

const std::vector<std::string>& getAllOpNames();
bool isValidOpName(const std::string& name);
OpCategory getOpCategory(const std::string& name);

} // namespace tptir

#endif // TPTIR_DIALECT_TPTIROPS_H
