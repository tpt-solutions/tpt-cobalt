#ifndef TPTIR_PASS_TPTIRPASSES_H
#define TPTIR_PASS_TPTIRPASSES_H
#include "../Dialect/TPTIRDialect.h"
#include "../Dialect/TPTIROps.h"
#include <unordered_set>
namespace tptir {
class Pass {
public:
  explicit Pass(const std::string& n) : name_(n) {}
  virtual ~Pass() = default;
  const std::string& name() const { return name_; }
  virtual bool run(Region* region) = 0;
  virtual bool runOnBlock(Block* block);
  virtual std::string statistics() const { return ""; }
private:
  std::string name_;
};
class PassPipeline {
public:
  PassPipeline() = default;
  ~PassPipeline();
  void addPass(Pass* pass);
  size_t run(Region* region);
  const std::vector<Pass*>& passes() const { return passes_; }
  size_t numPasses() const { return passes_.size(); }
  void clear();
private:
  std::vector<Pass*> passes_;
};
class CanonicalizePass : public Pass {
public:
  CanonicalizePass() : Pass("canonicalize") {}
  bool run(Region* region) override;
  std::string statistics() const override;
private:
  size_t foldCount_ = 0;
  bool foldIdentityOps(Block* block);
  bool removeUnreachableBlocks(Region* region);
};
class DeadCodeEliminationPass : public Pass {
public:
  DeadCodeEliminationPass() : Pass("dce") {}
  bool run(Region* region) override;
  std::string statistics() const override;
private:
  size_t removedOps_ = 0;
  void markUsedOps(Block*, std::unordered_set<Operation*>&);
  bool sweepUnusedOps(Block*, const std::unordered_set<Operation*>&);
};
class ConstantFolderPass : public Pass {
public:
  ConstantFolderPass() : Pass("const-fold") {}
  bool run(Region* region) override;
  std::string statistics() const override;
private:
  size_t foldedOps_ = 0;
  Value* tryFold(Operation* op);
  bool foldBlock(Block* block);
};
class VectorizePass : public Pass {
public:
  VectorizePass() : Pass("vectorize") {}
  bool run(Region* region) override;
  std::string statistics() const override;
private:
  size_t vectorizedOps_ = 0;
  bool vectorizeLoops(Region*);
};
class TensorLoweringPass : public Pass {
public:
  TensorLoweringPass() : Pass("tensor-lower") {}
  bool run(Region* region) override;
  std::string statistics() const override;
private:
  size_t loweredOps_ = 0;
  bool lowerTensorOps(Block*);
  bool lowerMMA(Operation*, Block*, size_t);
};
PassPipeline* createDefaultPassPipeline();
PassPipeline* createMinimalPassPipeline();
std::vector<std::string> getAvailablePassNames();
}
#endif
