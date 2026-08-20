#include "../../include/tptir/Dialect/TPTIRTypes.h"
namespace tptir {
bool typesEqual(const Type* a, const Type* b) {
  if (a == b) return true; if (!a || !b) return false;
  if (a->kind() != b->kind()) return false;
  if (a->isPrimitive()) return true;
  if (a->isTensor()) {
    auto ta = static_cast<const TensorType*>(a), tb = static_cast<const TensorType*>(b);
    if (ta->rank() != tb->rank()) return false;
    for (size_t i = 0; i < ta->rank(); i++) if (ta->getDim(i) != tb->getDim(i)) return false;
    return ta->addressSpace() == tb->addressSpace() && typesEqual(ta->elementType(), tb->elementType());
  }
  if (a->isVector()) {
    auto va = static_cast<const VectorType*>(a), vb = static_cast<const VectorType*>(b);
    return va->lanes() == vb->lanes() && typesEqual(va->elementType(), vb->elementType());
  }
  if (a->isMemRef()) {
    auto ma = static_cast<const MemRefType*>(a), mb = static_cast<const MemRefType*>(b);
    return ma->addressSpace() == mb->addressSpace() && typesEqual(ma->elementType(), mb->elementType());
  }
  return false;
}
Type* cloneType(const Type* t) { return t ? t->clone() : nullptr; }
void deleteType(Type* t) { delete t; }
void deleteTypes(std::vector<Type*>& types) { for (auto* t : types) delete t; types.clear(); }
int64_t shapeNumElements(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 0; int64_t n = 1;
  for (auto d : shape) { if (d < 0) return -1; n *= d; } return n;
}
bool isStaticShape(const std::vector<int64_t>& shape) { for (auto d : shape) if (d < 0) return false; return true; }
std::string shapeToString(const std::vector<int64_t>& shape) {
  std::string s; for (size_t i = 0; i < shape.size(); i++) { if (i > 0) s += "x"; s += shape[i] < 0 ? "*" : std::to_string(shape[i]); } return s;
}
Type* parseType(const std::string& ts) {
  if (ts == "i1") return I1Type(); if (ts == "i8") return I8Type();
  if (ts == "i16") return I16Type(); if (ts == "i32") return I32Type();
  if (ts == "i64") return I64Type(); if (ts == "f16") return F16Type();
  if (ts == "bf16") return BF16Type(); if (ts == "f32") return F32Type();
  if (ts == "f64") return F64Type(); if (ts == "index") return IndexType();
  if (ts.find("tensor<") == 0 || ts.find("memref<") == 0 || ts.find("vector<") == 0) {
    auto inner = ts.substr(ts.find('<') + 1, ts.size() - ts.find('<') - 2);
    auto xpos = inner.rfind('x');
    if (xpos == std::string::npos) return nullptr;
    std::vector<int64_t> shape; size_t start = 0, end; std::string shapeStr = inner.substr(0, xpos);
    while ((end = shapeStr.find('x', start)) != std::string::npos) { shape.push_back(std::stoll(shapeStr.substr(start, end - start))); start = end + 1; }
    if (start < shapeStr.size()) shape.push_back(std::stoll(shapeStr.substr(start)));
    AddressSpace as = AddressSpace::Global; std::string elStr = inner.substr(xpos + 1);
    auto cp = elStr.find(", "); if (cp != std::string::npos) { as = stringToAddressSpace(elStr.substr(cp + 2)); elStr = elStr.substr(0, cp); }
    Type* el = parseType(elStr); if (!el) return nullptr;
    if (ts.find("tensor<") == 0) return new TensorType(shape, el, as);
    if (ts.find("memref<") == 0) return new MemRefType(shape, el, as);
    return new VectorType(static_cast<uint32_t>(std::stoul(shapeStr)), el);
  }
  return nullptr;
}
FunctionType* parseFunctionType(const std::string& typeStr) {
  auto arrowPos = typeStr.find(" -> ");
  if (arrowPos == std::string::npos) return nullptr;
  auto inStr = typeStr.substr(1, arrowPos - 1);
  auto outStr = typeStr.substr(arrowPos + 4, typeStr.size() - arrowPos - 5);
  auto split = [](const std::string& s) {
    std::vector<Type*> r;
    size_t start = 0, end;
    while ((end = s.find(", ", start)) != std::string::npos) {
      r.push_back(parseType(s.substr(start, end - start)));
      start = end + 2;
    }
    if (start < s.size()) r.push_back(parseType(s.substr(start)));
    return r;
  };
  return new FunctionType(split(inStr), split(outStr));
}
}
