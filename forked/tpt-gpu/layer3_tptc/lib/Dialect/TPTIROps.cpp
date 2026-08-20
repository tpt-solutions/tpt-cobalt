#include "../../include/tptir/Dialect/TPTIROps.h"
namespace tptir {
Value::Value(Type* type_t id) : type_(type), id_(id) {}
std::string Value::toString() const {
  return "%" + std::to_string(id_) + (type_ ? " : " + type_->toString() : "");
}
Operation::Operation(OpId opId, std::vector<Value*> ops, Type* rt)
  : opId_(std::move(opId)), operands_(std::move(ops)), resultType_(rt), result_(nullptr) {}
Operation::~Operation() { delete resultType_; }
void Operation::setAttr(const std::string& k, const std::string& v) { attrs_[k] = v; }
std::string Operation::getAttr(const std::string& k, const std::string& d) const {
  auto it = attrs_.find(k); return it != attrs_.end() ? it->second : d;
}
bool Operation::hasAttr(const std::string& k) const { return attrs_.find(k) != attrs_.end(); }
std::string Operation::toString() const {
  std::string s;
  if (result_) s += result_->toString() + " = ";
  s += "\"" + name() + "\"(";
  for (size_t i = 0; i < operands_.size(); i++) {
    if (i > 0) s += ", ";
    if (operands_[i]) s += operands_[i]->toString();
  }
  s += ")";
  if (resultType_) s += " : " + resultType_->toString();
  return s;
}
Operation* Operation::clone() const {
  std::vector<Value*> ops; // shallow copy
  for (auto* op : operands_) ops.push_back(op);
  auto* n = new Operation(opId_, ops, resultType_ ? resultType_->clone() : nullptr);
  n->attrs_ = attrs_;
  return n;
}
Block::Block(const std::string& label) : label_(label) {}
Block::~Block() { for (auto* op : ops_) delete op; for (auto* v : args_) delete v; }
void Block::addOperation(Operation* op) { ops_.push_back(op); }
Operation* Block::getOperation(size_t i) const { return i < ops_.size() ? ops_[i] : nullptr; }
Operation* Block::getTerminator() const {
  return ops_.empty() ? nullptr : ops_.back();
}
void Block::addArgument(Value* arg) { args_.push_back(arg); }
std::string Block::toString() const {
  std::string s = "^" + label_ + ":\n";
  for (auto* op : ops_) s += "  " + op->toString() + "\n";
  return s;
}
Region::~Region() { for (auto* b : blocks_) delete b; }
void Region::addBlock(Block* block) { blocks_.push_back(block); }
Block* Region::getBlock(size_t i) const { return i < blocks_.size() ? blocks_[i] : nullptr; }
std::string Region::toString() const {
  std::string s;
  for (auto* b : blocks_) s += b->toString();
  return s;
}
static const std::vector<std::string>& allOps() {
  static const std::vector<std::string> ops = {
    "tptir.addi","tptir.subi","tptir.muli","tptir.divi",
    "tptir.addf","tptir.subf","tptir.mulf","tptir.divf","tptir.fma",
    "tptir.andi","tptir.ori","tptir.xori","tptir.slli","tptir.srli","tptir.srai",
    "tptir.cmpeq","tptir.cmpne","tptir.cmplt","tptir.cmpgt","tptir.cmple","tptir.cmpge",
    "tptir.load","tptir.store","tptir.alloc","tptir.dealloc",
    "tptir.tensor_load","tptir.tensor_store","tptir.mma","tptir.contraction",
    "tptir.vector_load","tptir.vector_store","tptir.vector_add",
    "tptir.warp_shuffle","tptir.warp_reduce",
    "tptir.br","tptir.cond_br","tptir.return","tptir.call",
    "tptir.fptosi","tptir.sitofp","tptir.extsi","tptir.trunci",
    "tptir.sync","tptir.fence","tptir.predicate","tptir.constant"
  };
  return ops;
}
const std::vector<std::string>& getAllOpNames() { return allOps(); }
bool isValidOpName(const std::string& name) {
  for (auto& op : allOps()) if (op == name) return true;
  return false;
}
OpCategory getOpCategory(const std::string& name) {
  if (name.find("add") != std::string::npos || name.find("sub") != std::string::npos ||
      name.find("mul") != std::string::npos || name.find("div") != std::string::npos ||
      name.find("fma") != std::string::npos || name.find("neg") != std::string::npos)
    return OpCategory::Arithmetic;
  if (name.find("and") != std::string::npos || name.find("or") != std::string::npos ||
      name.find("xor") != std::string::npos || name.find("shl") != std::string::npos ||
      name.find("shr") != std::string::npos)
    return OpCategory::Logical;
  if (name.find("cmp") != std::string::npos) return OpCategory::Comparison;
  if (name.find("load") != std::string::npos || name.find("store") != std::string::npos ||
      name.find("alloc") != std::string::npos)
    return OpCategory::Memory;
  if (name.find("tensor") != std::string::npos || name.find("mma") != std::string::npos ||
      name.find("contraction") != std::string::npos)
    return OpCategory::Tensor;
  if (name.find("vector") != std::string::npos || name.find("warp") != std::string::npos)
    return OpCategory::VectorSIMT;
  if (name.find("br") != std::string::npos || name.find("return") != std::string::npos ||
      name.find("call") != std::string::npos)
    return OpCategory::ControlFlow;
  if (name.find("fptosi") != std::string::npos || name.find("sitofp") != std::string::npos)
    return OpCategory::Conversion;
  if (name.find("sync") != std::string::npos || name.find("fence") != std::string::npos)
    return OpCategory::Special;
  if (name.find("constant") != std::string::npos) return OpCategory::Constant;
  return OpCategory::Other;
}
}
