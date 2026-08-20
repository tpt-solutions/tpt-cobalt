// =============================================================================
// TPTIRTypes.h — TPTIR Type System Extensions
// =============================================================================
// TPT GPU — Tensor Processing Technology
// License: Apache License 2.0 (with Express Patent Grant)
// =============================================================================

#ifndef TPTIR_DIALECT_TPTIRTYPES_H
#define TPTIR_DIALECT_TPTIRTYPES_H

#include "TPTIRDialect.h"
#include <vector>
#include <cstdint>

namespace tptir {

bool typesEqual(const Type* a, const Type* b);
Type* cloneType(const Type* t);
void deleteType(Type* t);
void deleteTypes(std::vector<Type*>& types);

int64_t shapeNumElements(const std::vector<int64_t>& shape);
bool isStaticShape(const std::vector<int64_t>& shape);
std::string shapeToString(const std::vector<int64_t>& shape);

Type* parseType(const std::string& typeStr);
FunctionType* parseFunctionType(const std::string& typeStr);

} // namespace tptir

#endif // TPTIR_DIALECT_TPTIRTYPES_H
