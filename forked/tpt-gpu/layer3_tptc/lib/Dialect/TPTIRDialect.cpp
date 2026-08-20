#include "../../include/tptir/Dialect/TPTIRDialect.h"
#include <unordered_map>
namespace tptir {
static bool g_registered = false;
const char* addressSpaceToString(AddressSpace as) {
  switch(as) {
    case AddressSpace::Global: return "global";
    case AddressSpace::Shared: return "shared";
    case AddressSpace::Local: return "local";
    case AddressSpace::Constant: return "constant";
    case AddressSpace::Generic: return "generic";
  } return "unknown";
}
AddressSpace stringToAddressSpace(const std::string& str) {
  static const std::unordered_map<std::string,AddressSpace> m={
    {"global",AddressSpace::Global},{"shared",AddressSpace::Shared},
    {"local",AddressSpace::Local},{"constant",AddressSpace::Constant},
    {"generic",AddressSpace::Generic}};
  auto it=m.find(str); return it!=m.end()?it->second:AddressSpace::Global;
}
uint32_t getTypeBitWidth(TypeKind kind) {
  switch(kind) {
    case TypeKind::I1:return 1;case TypeKind::I8:return 8;
    case TypeKind::I16:return 16;case TypeKind::I32:return 32;
    case TypeKind::I64:return 64;case TypeKind::F16:return 16;
    case TypeKind::BF16:return 16;case TypeKind::F32:return 32;
    case TypeKind::F64:return 64;case TypeKind::Index:return 32;
    default:return 0;
  }
}
std::string PrimitiveType::toString() const {
  switch(kind()) {
    case TypeKind::I1:return "i1";case TypeKind::I8:return "i8";
    case TypeKind::I16:return "i16";case TypeKind::I32:return "i32";
    case TypeKind::I64:return "i64";case TypeKind::F16:return "f16";
    case TypeKind::BF16:return "bf16";case TypeKind::F32:return "f32";
    case TypeKind::F64:return "f64";case TypeKind::Index:return "index";
    default:return "unknown";
  }
}
TensorType::TensorType(std::vector<int64_t> s,Type* e,AddressSpace a)
  :Type(TypeKind::Tensor),shape_(std::move(s)),elType_(e),as_(a){}
int64_t TensorType::numElements() const {
  if(shape_.empty()) return 0; int64_t n=1;
  for(auto d:shape_){if(d<0)return -1;n*=d;} return n;
}
std::string TensorType::toString() const {
  std::string s="tensor<";
  for(size_t i=0;i<shape_.size();i++){
    if(i>0)s+="x"; s+=shape_[i]<0?"*":std::to_string(shape_[i]);}
  s+=elType_->toString();
  if(as_!=AddressSpace::Global)s+=", "+std::string(addressSpaceToString(as_));
  return s+">";
}
TensorType* TensorType::clone() const{return new TensorType(shape_,elType_->clone(),as_);}
VectorType::VectorType(uint32_t l,Type* e):Type(TypeKind::Vector),lanes_(l),elType_(e){}
std::string VectorType::toString() const{return "vector<"+std::to_string(lanes_)+"x"+elType_->toString()+">";}
VectorType* VectorType::clone() const{return new VectorType(lanes_,elType_->clone());}
MemRefType::MemRefType(std::vector<int64_t> s,Type* e,AddressSpace a)
  :Type(TypeKind::MemRef),shape_(std::move(s)),elType_(e),as_(a){}
std::string MemRefType::toString() const {
  std::string s="memref<";
  for(size_t i=0;i<shape_.size();i++){
    if(i>0)s+="x"; s+=shape_[i]<0?"*":std::to_string(shape_[i]);}
  s+=elType_->toString();
  if(as_!=AddressSpace::Global)s+=", "+std::string(addressSpaceToString(as_));
  return s+">";
}
MemRefType* MemRefType::clone() const{return new MemRefType(shape_,elType_->clone(),as_);}
FunctionType::FunctionType(std::vector<Type*> i,std::vector<Type*> o)
  :Type(TypeKind::Function),inputs_(std::move(i)),outputs_(std::move(o)){}
FunctionType::~FunctionType(){for(auto*t:inputs_)delete t;for(auto*t:outputs_)delete t;}
std::string FunctionType::toString() const {
  std::string s="(";
  for(size_t i=0;i<inputs_.size();i++){if(i>0)s+=", ";s+=inputs_[i]->toString();}
  s+=") -> (";
  for(size_t i=0;i<outputs_.size();i++){if(i>0)s+=", ";s+=outputs_[i]->toString();}
  return s+")";
}
FunctionType* FunctionType::clone() const {
  std::vector<Type*>in,out;
  for(auto*t:inputs_)in.push_back(t->clone());
  for(auto*t:outputs_)out.push_back(t->clone());
  return new FunctionType(std::move(in),std::move(out));
}
void registerTPTIRDialect(){g_registered=true;}
bool isTPTIRDialectRegistered(){return g_registered;}
}
