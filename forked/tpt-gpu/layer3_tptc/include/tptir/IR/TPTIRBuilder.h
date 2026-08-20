#ifndef TPTIR_IR_TPTIRBUILDER_H
#define TPTIR_IR_TPTIRBUILDER_H
#include "../Dialect/TPTIRDialect.h"
#include "../Dialect/TPTIROps.h"
#include <stack>
namespace tptir {
class IRBuilder {
public:
  IRBuilder();
  void setInsertionPoint(Block* block);
  Block* currentBlock() const { return currentBlock_; }
  Region* currentRegion() const { return currentRegion_; }
  void setCurrentRegion(Region* region) { currentRegion_ = region; }
  Value* createAddi(Value* lhs, Value* rhs);
  Value* createSubi(Value* lhs, Value* rhs);
  Value* createMuli(Value* lhs, Value* rhs);
  Value* createAddf(Value* lhs, Value* rhs);
  Value* createSubf(Value* lhs, Value* rhs);
  Value* createMulf(Value* lhs, Value* rhs);
  Value* createFMA(Value* a, Value* b, Value* c);
  Value* createAnd(Value* lhs, Value* rhs);
  Value* createOr(Value* lhs, Value* rhs);
  Value* createXor(Value* lhs, Value* rhs);
  Value* createCmpEQ(Value* lhs, Value* rhs);
  Value* createCmpLT(Value* lhs, Value* rhs);
  Value* createConstantI32(int32_t value);
  Value* createConstantF32(float value);
  Value* createLoad(Value* memref, Type* resultType);
  void createStore(Value* value, Value* memref);
  Value* createTensorLoad(Value* memref, Type* tensorType);
  void createTensorStore(Value* tensor, Value* memref);
  Value* createMMA(Value* a, Value* b, Value* c, Type* resultType);
  Value* createVectorLoad(Value* memref, Value* index, Type* vecType);
  void createVectorStore(Value* vector, Value* memref, Value* index);
  Value* createVectorAdd(Value* lhs, Value* rhs, Type* vecType);
  void createBranch(Block* target);
  void createCondBranch(Value* condition, Block* trueBlock, Block* falseBlock);
  void createReturn(const std::vector<Value*>& values = {});
  Block* createBlock(const std::string& label = "");
  void pushBlock(Block* block);
  void popBlock();
  Value* makeValue(Type* type);
private:
  Block* currentBlock_{nullptr};
  Region* currentRegion_{nullptr};
  uint64_t nextValueId_{0};
  std::stack<Block*> blockStack_;
};
}
#endif
