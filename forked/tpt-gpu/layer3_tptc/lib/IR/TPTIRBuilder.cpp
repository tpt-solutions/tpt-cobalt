#include "../../include/tptir/IR/TPTIRBuilder.h"
namespace tptir {
IRBuilder::IRBuilder():currentBlock_(nullptr),currentRegion_(nullptr),nextValueId_(0){}
void IRBuilder::setInsertionPoint(Block* b){currentBlock_=b;}
Value* IRBuilder::makeValue(Type* t){return new Value(t,nextValueId_++);}
Block* IRBuilder::createBlock(const std::string& l){
  auto* b=new Block(l.empty()?"bb"+std::to_string(nextValueId_++):l);
  if(currentRegion_)currentRegion_->addBlock(b);return b;
}
void IRBuilder::pushBlock(Block* b){blockStack_.push(currentBlock_);currentBlock_=b;}
void IRBuilder::popBlock(){if(!blockStack_.empty()){currentBlock_=blockStack_.top();blockStack_.pop();}}
#define BINOP(n,on) Value* IRBuilder::create##n(Value* l,Value* r){Type* rt=l->type();auto* op=createArithOp("tptir." on,l,r,rt);auto* v=makeValue(rt);op->setResult(v);if(currentBlock_)currentBlock_->addOperation(op);return v;}
BINOP(Addi,"addi")BINOP(Subi,"subi")BINOP(Muli,"muli")
BINOP(Addf,"addf")BINOP(Subf,"subf")BINOP(Mulf,"mulf")
BINOP(And,"andi")BINOP(Or,"ori")BINOP(Xor,"xori")
Value* IRBuilder::createFMA(Value* a,Value* b,Value* c){Type* rt=a->type();auto* op=createArithOp("tptir.fma",a,b,rt);auto* v=makeValue(rt);op->setResult(v);op->setAttr("c",c?c->toString():"");if(currentBlock_)currentBlock_->addOperation(op);return v;}
Value* IRBuilder::createCmpEQ(Value* l,Value* r){auto* rt=I1Type();auto* op=createArithOp("tptir.cmpeq",l,r,rt);auto* v=makeValue(rt);op->setResult(v);if(currentBlock_)currentBlock_->addOperation(op);return v;}
Value* IRBuilder::createCmpLT(Value* l,Value* r){auto* rt=I1Type();auto* op=createArithOp("tptir.cmplt",l,r,rt);auto* v=makeValue(rt);op->setResult(v);if(currentBlock_)currentBlock_->addOperation(op);return v;}
Value* IRBuilder::createConstantI32(int32_t v){auto* rt=I32Type();auto* op=createConstantOp(rt,std::to_string(v));auto* rv=makeValue(rt);op->setResult(rv);if(currentBlock_)currentBlock_->addOperation(op);return rv;}
Value* IRBuilder::createConstantF32(float v){auto* rt=F32Type();auto* op=createConstantOp(rt,std::to_string(v));auto* rv=makeValue(rt);op->setResult(rv);if(currentBlock_)currentBlock_->addOperation(op);return rv;}
Value* IRBuilder::createLoad(Value* m,Type* rt){auto* op=createLoadOp(m,rt);auto* v=makeValue(rt);op->setResult(v);if(currentBlock_)currentBlock_->addOperation(op);return v;}
void IRBuilder::createStore(Value* v,Value* m){if(currentBlock_)currentBlock_->addOperation(createStoreOp(v,m));}
void IRBuilder::createBranch(Block* t){if(currentBlock_)currentBlock_->addOperation(createBranchOp(t));}
void IRBuilder::createCondBranch(Value* c,Block* t,Block* f){if(currentBlock_)currentBlock_->addOperation(createCondBranchOp(c,t,f));}
void IRBuilder::createReturn(const std::vector<Value*>& v){if(currentBlock_)currentBlock_->addOperation(createReturnOp(v));}
Value* IRBuilder::createMMA(Value* a,Value* b,Value* c,Type* rt){auto* op=createMMAOp(a,b,c,rt);auto* v=makeValue(rt);op->setResult(v);if(currentBlock_)currentBlock_->addOperation(op);return v;}
}
