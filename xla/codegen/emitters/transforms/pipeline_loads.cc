/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-W#pragma-messages"

#include "xla/codegen/emitters/transforms/vectorization_and_pipelining_utils.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_PIPELINELOADS
#include "xla/codegen/emitters/transforms/passes.h.inc"

using ::mlir::Value;
using ::mlir::AffineExpr;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::OpFoldResult;

namespace scf = ::mlir::scf;
namespace arith = ::mlir::arith;

Value FindThreadIdx(ApplyIndexingOp op) {
  if (op.getIndexingMap().GetRangeVarsCount() != 0) {
    // The apply_indexing should have been canonicalized; there should
    // be no symbols left.
    return Value();
  }
  for (int i = 0; i < op.getIndexingMap().GetDimVarsCount(); ++i) {
    Value val = op.getOperand(i);
    if (mlir::isa<::mlir::gpu::ThreadIdOp>(val.getDefiningOp())) {
      // TODO: We don't generate any dims except x, but this needs to be verified here.
      return val;
    }
  }
  // Nothing found.
  return Value();
}

std::optional<int64_t> GetPipelinedTransactionSize(
    const IndexingMap& indexing_map,
    const std::map<int64_t, mlir::AffineExpr>& stride_to_sym) {
  int64_t expected_stride = 1;
  for (auto [stride, sym] : stride_to_sym) {
    if (stride != expected_stride) {
      return std::nullopt;
    }
    Interval bound = indexing_map.GetDimensionBound(mlir::cast<mlir::AffineDimExpr>(sym).getPosition());
    if (bound.lower != 0) {
      return std::nullopt;
    }
    expected_stride *= bound.upper + 1;
  }

  return expected_stride;
}

// TODO: unify with GetVectorType in vectorize_loads_stores.
std::optional<int64_t> GetTripCount(scf::ForOp loop){
  if (mlir::getConstantIntValue(loop.getStep()) != 1 ||
      mlir::getConstantIntValue(loop.getLowerBound()) != 0) {
    return std::nullopt;
  }
  return
      mlir::getConstantIntValue(loop.getUpperBound());
}

struct PipelineTensorExtract
    : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  // TODO: yikes. Refactor this into something understandable.
  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (mlir::isa_and_present<gpu::DequeueOp>(op.getTensor().getDefiningOp())) {
      return rewriter.notifyMatchFailure(op, "already pipelined");
    }

    auto loop = mlir::dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!loop) {
      std::cerr << "not pipelineable\n";
      return rewriter.notifyMatchFailure(op, "not a pipelineable load");
    }

    if (op.getIndices().size() != 1) {
      std::cerr << "not 1d\n";
      return rewriter.notifyMatchFailure(op, "not a 1D extract");
    }
    auto index = op.getIndices().front();

    if (index == loop.getInductionVar()) {
      // TODO: implement this case.
      std::cerr << "induction var only index\n";
      return rewriter.notifyMatchFailure(op, "TODO: Not implemented yet");
    }
    auto apply_indexing = mlir::dyn_cast_or_null<ApplyIndexingOp>(index.getDefiningOp());
    if (!apply_indexing) {
      std::cerr << "no apply_indexing\n";
      return rewriter.notifyMatchFailure(op, "not a pipelineable load");
    }
    IndexingMap indexing_map = apply_indexing.getIndexingMap();

    auto strides = GetStrides(indexing_map.GetAffineMap().getResult(0));
    llvm::DenseMap<mlir::Value, mlir::AffineExpr> val_to_expr;
    for (int i = 0; i < apply_indexing.getNumOperands(); ++i) {
      auto expr = mlir::getAffineDimExpr(i, apply_indexing.getContext());
      if (!val_to_expr.try_emplace(apply_indexing.getOperand(i), expr).second) {
        // TODO: check if this is actually possible. If it is, we need a canonicalization pattern for it.
        std::cerr << "operands not mutually exclusive\n";
        return rewriter.notifyMatchFailure(op, "operands not mutually exclusive");
      }
    }

    Value thread_idx = FindThreadIdx(apply_indexing);
    if (!thread_idx) {
      std::cerr << "no thread idx\n";
      return rewriter.notifyMatchFailure(op, "no thread idx found");
    }

    mlir::AffineDimExpr thread_idx_expr = mlir::cast<mlir::AffineDimExpr>(val_to_expr[thread_idx]);
    std::optional<int64_t> stride = strides[thread_idx_expr];
    if (!stride.has_value()) {
      std::cerr << "no stride for thread idx\n";
      return rewriter.notifyMatchFailure(op,
          "unable to get stride of thread index in load");
    }

    std::map<int64_t, AffineExpr> stride_to_sym;
    stride_to_sym[*stride] = thread_idx_expr;

    llvm::SmallVector<Value> load_operands(indexing_map.GetDimVarsCount());
    load_operands[thread_idx_expr.getPosition()] = thread_idx;
    AffineExpr load_offset = thread_idx_expr * *stride;

    while (loop) {
      if (!IsConflictFree(op, loop)) {
        std::cerr << "not conflict free\n";
        return rewriter.notifyMatchFailure(op,
            "source may be written in the loop");
      }

      auto trip_count = GetTripCount(loop);
      if (!trip_count || trip_count < 2) {
        return rewriter.notifyMatchFailure(op, "loop trip count unknown or too small");
      }

      // Check if we can pipeline.
      if (auto tx_size = GetPipelinedTransactionSize(indexing_map,
                                                     stride_to_sym)) {
        // Check that the remaining operands are all defined outside the loop.
        int64_t induction_var_pos;
        for (auto [val, expr] : val_to_expr) {
          llvm::errs() << "checking operand " << val << " for expr " << expr << "\n";
          if (val == loop.getInductionVar()) {
            induction_var_pos = mlir::cast<mlir::AffineDimExpr>(expr).getPosition();
          } else if (!val.getParentRegion()->isProperAncestor(
                  &loop.getBodyRegion())) {
            std::cerr << "operand for d" << mlir::cast<mlir::AffineDimExpr>(expr).getPosition() << " may not be constant\n";
            return rewriter.notifyMatchFailure(
                op, "operand may not be constant");
          }
        }

        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        b.setInsertionPoint(loop);
        auto zero = b.create<arith::ConstantIndexOp>(0);
        auto one = b.create<arith::ConstantIndexOp>(1);
        llvm::SmallVector<mlir::Value> new_operands(apply_indexing.getOperands());
        for (auto [_, expr] : stride_to_sym) {
          new_operands[mlir::cast<mlir::AffineDimExpr>(expr).getPosition()] = zero;
        }
        
        Value is_leader = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, thread_idx, zero);

        auto tx_ty = op.getTensor().getType().clone(llvm::SmallVector<int64_t>{*tx_size});
        auto empty_pipe_ty =  gpu::SharedMemoryPipeType::get(
                b.getContext(),
                tx_ty,
                /*capacity=*/2, // TODO: always 2?
                /*level=*/0);
        Value pipe = b.create<gpu::AllocatePipeOp>(
            empty_pipe_ty,
            is_leader,
            128 /* TODO: get from domain */);

        for (auto idx : {zero, one}) {
          new_operands[induction_var_pos] = idx;
          Value base_index = b.create<ApplyIndexingOp>(new_operands, indexing_map).getResult(0);
          llvm::SmallVector<OpFoldResult> offsets {base_index};
          llvm::SmallVector<OpFoldResult> sizes {b.getIndexAttr(*tx_size)};
          llvm::SmallVector<OpFoldResult> strides {b.getIndexAttr(1)};
          auto slice = b.create<mlir::tensor::ExtractSliceOp>(
              tx_ty, op.getTensor(), offsets, sizes, strides);
          pipe = b.create<gpu::EnqueueOp>(pipe, slice);
        }

        Value dequeued;
        auto new_for = loop.replaceWithAdditionalYields(
            rewriter, pipe,
            /*replaceInitOperandUsesInLoop=*/false,
            [&](mlir::OpBuilder& yield_b, mlir::Location yield_loc,
                llvm::ArrayRef<mlir::BlockArgument> bbarg) -> llvm::SmallVector<Value> {
              mlir::ImplicitLocOpBuilder dequeue_builder(yield_loc, yield_b);
              dequeue_builder.setInsertionPointToStart(loop.getBody());
              Value orig_pipe = bbarg[0];
              // TODO: type inference
              auto value_and_pipe = dequeue_builder.create<gpu::DequeueOp>(
                  empty_pipe_ty.getElementType(), empty_pipe_ty.cloneWithLevel(1),
                  orig_pipe).getResults();
              dequeued = value_and_pipe[0];
              Value pipe = value_and_pipe[1];

              auto induction_var =
                  mlir::cast<scf::ForOp>(bbarg[0].getOwner()->getParentOp())
                 .getInductionVar();
              auto two = yield_b.create<arith::ConstantIndexOp>(yield_loc, 2);
              auto induction_var_plus_two = yield_b.create<arith::AddIOp>(yield_loc, induction_var, two);
              Value cond = yield_b.create<arith::CmpIOp>(
                  yield_loc, arith::CmpIPredicate::ult,
                  induction_var_plus_two, yield_b.create<arith::ConstantIndexOp>(yield_loc, *trip_count));
              auto enqueue = yield_b.create<scf::IfOp>(
                  yield_loc, cond,
                  /*thenBuilder=*/
                  [&](mlir::OpBuilder& then_b, mlir::Location then_loc) {
                    new_operands[induction_var_pos] = induction_var_plus_two;
                    Value base_index = then_b.create<ApplyIndexingOp>(then_loc, new_operands, indexing_map).getResult(0);
                    llvm::SmallVector<OpFoldResult> offsets {base_index};
                    llvm::SmallVector<OpFoldResult> sizes {then_b.getIndexAttr(*tx_size)};
                    llvm::SmallVector<OpFoldResult> strides {then_b.getIndexAttr(1)};
                    auto slice = then_b.create<mlir::tensor::ExtractSliceOp>(
                        then_loc, tx_ty, op.getTensor(), offsets, sizes, strides);
                    Value new_pipe = then_b.create<gpu::EnqueueOp>(then_loc, pipe, slice);
                    then_b.create<scf::YieldOp>(then_loc, new_pipe);
                  },
                  /*elseBuilder=*/
                  [&](mlir::OpBuilder& else_b, mlir::Location else_loc) {
                    Value new_pipe = else_b.create<gpu::EnqueueUndefOp>(else_loc, pipe);
                    else_b.create<scf::YieldOp>(else_loc, new_pipe);
                  });

              return {enqueue.getResult(0)};
            });

        rewriter.setInsertionPoint(op);
        mlir::AffineMap load_affine_map = mlir::AffineMap::get(indexing_map.GetDimVarsCount(), 0, load_offset);
        IndexingMap load_indexing_map(load_affine_map, indexing_map.GetDimVars(), {}, {});
        for (auto& operand : load_operands) {
          if (!operand) operand = zero;
        }

        Value index = rewriter.create<ApplyIndexingOp>(op.getLoc(), load_operands, load_indexing_map).getResult(0);
        rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(
            op, dequeued, index);
        llvm::errs()  << *pipe.getDefiningOp()->getParentRegion()->begin() << "\n";
        return llvm::success();
      } else {
        std::cerr << "not contiguous\n";
      }

      AffineExpr loop_var_expr = val_to_expr[loop.getInductionVar()];
      if (!loop_var_expr) { 
        // e.g. broadcast. Just skip this for now.
        // TODO: double check if it could be usefully pipelined.
        std::cerr << "induction var unused\n";
        return rewriter.notifyMatchFailure(op,
            "loop induction variable unused in apply_indexing");
      }
      // TODO: don't mutate val_to_expr.
      val_to_expr.erase(loop.getInductionVar());

      stride = strides[loop_var_expr];
      if (!stride.has_value()) {
        std::cerr << "no stride for loop var\n";
        return rewriter.notifyMatchFailure(op,
            "unable to get stride of loop variable in load");
      }

      if (!stride_to_sym.try_emplace(*stride, loop_var_expr).second) {
        std::cerr << "duplicate stride\n";
        return rewriter.notifyMatchFailure(op, "duplicate stride");
      }
      
      load_operands[mlir::cast<mlir::AffineDimExpr>(loop_var_expr).getPosition()] = loop.getInductionVar();
      load_offset = load_offset + loop_var_expr * *stride;
      loop = mlir::dyn_cast_or_null<scf::ForOp>(loop->getParentOp());
    }

    std::cerr << "no loop found\n";
    return rewriter.notifyMatchFailure(op, "not a pipelineable load");
  }
};

class PipelineLoadsPass
    : public impl::PipelineLoadsBase<PipelineLoadsPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<PipelineTensorExtract>(mlir_context);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreatePipelineLoadsPass() {
  return std::make_unique<PipelineLoadsPass>();
}

}  // namespace emitters
}  // namespace xla
