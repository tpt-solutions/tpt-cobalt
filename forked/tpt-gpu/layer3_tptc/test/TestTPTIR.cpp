#include "../include/tptir/Dialect/TPTIRDialect.h"
#include "../include/tptir/Dialect/TPTIRTypes.h"
#include "../include/tptir/Dialect/TPTIROps.h"
#include "../include/tptir/IR/TPTIRBuilder.h"
#include "../include/tptir/Pass/TPTIRPasses.h"
#include "../include/tptir/CodeGen/TPTCodeGen.h"
#include <cassert>
#include <iostream>

int main() {
  tptir::registerTPTIRDialect();
  assert(tptir::isTPTIRDialectRegistered());

  auto* i32 = tptir::I32Type();
  assert(i32->kind() == tptir::TypeKind::I32);
  assert(i32->toString() == "i32");

  auto* tensor = new tptir::TensorType({16, 16}, tptir::F16Type());
  assert(tensor->rank() == 2);
  assert(tensor->numElements() == 256);

  auto* vec = new tptir::VectorType(32, tptir::F32Type());
  assert(vec->lanes() == 32);

  auto* memref = new tptir::MemRefType({1024}, tptir::F32Type(), tptir::AddressSpace::Shared);
  assert(memref->rank() == 1);

  tptir::IRBuilder builder;
  auto* entry = builder.createBlock("entry");
  auto* region = new tptir::Region();
  region->addBlock(entry);
  builder.setCurrentRegion(region);
  builder.setInsertionPoint(entry);

  auto* c0 = builder.createConstantI32(0);
  auto* c1 = builder.createConstantI32(1);
  auto* sum = builder.createAddi(c0, c1);
  builder.createReturn({sum});

  auto* pipeline = tptir::createDefaultPassPipeline();
  pipeline->run(region);

  tptir::CodeGenOptions opts;
  opts.target = tptir::CodeGenTarget::TPTIRText;
  tptir::TPTCodeGen codegen(opts);
  std::string output = codegen.generate(region);
  assert(!output.empty());

  std::cout << "=== ALL C++ TESTS PASSED ===" << std::endl;
  std::cout << output << std::endl;

  delete pipeline;
  delete region;
  return 0;
}
