#include "../../include/tptir/CodeGen/TPTCodeGen.h"
namespace tptir {
TPTCodeGen::TPTCodeGen(const CodeGenOptions& o):options_(o),nextLabelId_(0){}
std::string TPTCodeGen::generate(Region* r){switch(options_.target){case CodeGenTarget::TPTISA:return emitTPTISA(r);case CodeGenTarget::LLVMIR:return emitLLVMIR(r);case CodeGenTarget::TPTIRText:return emitTPTIRText(r);}return"";}
std::string TPTCodeGen::generateFromBlocks(const std::vector<Block*>& b){Region r;for(auto*b2:b)r.addBlock(b2);return generate(&r);}
std::string TPTCodeGen::emitTPTISA(Region* r){std::stringstream ss;ss<<"; TPT ISA\n";for(auto*b:r->blocks())for(auto*op:b->operations())ss<<emitOperation(op,ss)<<"\n";return ss.str();}
std::string TPTCodeGen::emitLLVMIR(Region* r){std::stringstream ss;ss<<"; LLVM IR\ndefine void @kernel(){\n";for(auto*b:r->blocks())for(auto*op:b->operations())ss<<"  "<<emitOperation(op,ss)<<"\n";ss<<"}\n";return ss.str();}
std::string TPTCodeGen::emitTPTIRText(Region* r){return r->toString();}
std::string TPTCodeGen::emitOperation(const Operation* op,std::stringstream&){return op->toString();}
std::string TPTCodeGen::valueName(const Value* v)const{auto it=valueNames_.find(v);return it!=valueNames_.end()?it->second:v?"%v"+std::to_string(v->id()):"%undef";}
std::string TPTCodeGen::typeName(const Type* t)const{return t?t->toString():"none";}
}
