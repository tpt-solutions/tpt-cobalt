// =============================================================================
// TPTIRDialect.h — TPTIR MLIR-Compatible Dialect Definition
// =============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// =============================================================================

#ifndef TPTIR_DIALECT_TPTIRDIALECT_H
#define TPTIR_DIALECT_TPTIRDIALECT_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace tptir {

constexpr const char* kDialectNamespace = "tptir";

enum class AddressSpace : uint8_t {
  Global = 0, Shared = 1, Local = 2, Constant = 3, Generic = 4
};

const char* addressSpaceToString(AddressSpace as);
AddressSpace stringToAddressSpace(const std::string& str);

enum class TypeKind : uint8_t {
  I1=0, I8=1, I16=2, I32=3, I64=4,
  F16=5, BF16=6, F32=7, F64=8, Index=9,
  Function=10, Tensor=11, Vector=12, MemRef=13, None=14
};

uint32_t getTypeBitWidth(TypeKind kind);

class Type {
public:
  explicit Type(TypeKind kind) : kind_(kind) {}
  virtual ~Type() = default;
  TypeKind kind() const { return kind_; }
  bool isPrimitive() const { return kind_ >= TypeKind::I1 && kind_ <= TypeKind::Index; }
  bool isTensor() const { return kind_ == TypeKind::Tensor; }
  bool isVector() const { return kind_ == TypeKind::Vector; }
  bool isMemRef() const { return kind_ == TypeKind::MemRef; }
  bool isFunction() const { return kind_ == TypeKind::Function; }
  virtual std::string toString() const = 0;
  virtual Type* clone() const = 0;
private:
  TypeKind kind_;
};

class PrimitiveType : public Type {
public:
  explicit PrimitiveType(TypeKind kind) : Type(kind) {}
  std::string toString() const override;
  PrimitiveType* clone() const override { return new PrimitiveType(kind()); }
  uint32_t bitWidth() const { return getTypeBitWidth(kind()); }
};

class TensorType : public Type {
public:
  TensorType(std::vector<int64_t> shape, Type* elType, AddressSpace as = AddressSpace::Global);
  ~TensorType() override { delete elType_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  int64_t getDim(size_t i) const { return shape_[i]; }
  size_t rank() const { return shape_.size(); }
  int64_t numElements() const;
  Type* elementType() const { return elType_; }
  AddressSpace addressSpace() const { return as_; }
  std::string toString() const override;
  TensorType* clone() const override;
private:
  std::vector<int64_t> shape_;
  Type* elType_;
  AddressSpace as_;
};

class VectorType : public Type {
public:
  VectorType(uint32_t lanes, Type* elType);
  ~VectorType() override { delete elType_; }
  uint32_t lanes() const { return lanes_; }
  Type* elementType() const { return elType_; }
  std::string toString() const override;
  VectorType* clone() const override;
private:
  uint32_t lanes_;
  Type* elType_;
};

class MemRefType : public Type {
public:
  MemRefType(std::vector<int64_t> shape, Type* elType, AddressSpace as = AddressSpace::Global);
  ~MemRefType() override { delete elType_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  size_t rank() const { return shape_.size(); }
  Type* elementType() const { return elType_; }
  AddressSpace addressSpace() const { return as_; }
  std::string toString() const override;
  MemRefType* clone() const override;
private:
  std::vector<int64_t> shape_;
  Type* elType_;
  AddressSpace as_;
};

class FunctionType : public Type {
public:
  FunctionType(std::vector<Type*> inputs, std::vector<Type*> outputs);
  ~FunctionType() override;
  const std::vector<Type*>& inputs() const { return inputs_; }
  const std::vector<Type*>& outputs() const { return outputs_; }
  std::string toString() const override;
  FunctionType* clone() const override;
  size_t numInputs() const { return inputs_.size(); }
  size_t numOutputs() const { return outputs_.size(); }
private:
  std::vector<Type*> inputs_;
  std::vector<Type*> outputs_;
};

inline PrimitiveType* I1Type()   { return new PrimitiveType(TypeKind::I1); }
inline PrimitiveType* I8Type()   { return new PrimitiveType(TypeKind::I8); }
inline PrimitiveType* I16Type()  { return new PrimitiveType(TypeKind::I16); }
inline PrimitiveType* I32Type()  { return new PrimitiveType(TypeKind::I32); }
inline PrimitiveType* I64Type()  { return new PrimitiveType(TypeKind::I64); }
inline PrimitiveType* F16Type()  { return new PrimitiveType(TypeKind::F16); }
inline PrimitiveType* BF16Type(){ return new PrimitiveType(TypeKind::BF16); }
inline PrimitiveType* F32Type()  { return new PrimitiveType(TypeKind::F32); }
inline PrimitiveType* F64Type()  { return new PrimitiveType(TypeKind::F64); }
inline PrimitiveType* IndexType(){ return new PrimitiveType(TypeKind::Index); }

void registerTPTIRDialect();
bool isTPTIRDialectRegistered();

} // namespace tptir

#endif // TPTIR_DIALECT_TPTIRDIALECT_H
