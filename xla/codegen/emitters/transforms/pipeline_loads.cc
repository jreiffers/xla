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

std::optional<int64_t> GetPipelinedTransactionSize(
    const IndexingMap& indexing_map,
    int64_t thread_id_dim,
    int64_t thread_id_stride,
    llvm::ArrayRef<int64_t> loop_dims,
    llvm::ArrayRef<int64_t> loop_strides) {
  std::map<int64_t, int64_t> strides_to_dims;
  for (int i = 0; i < loop_dims.size(); ++i) {
    strides_to_dims[loop_strides[i]] = loop_dims[i];
  }
  strides_to_dims[thread_id_stride] = thread_id_dim;

  if (strides_to_dims.size() != loop_dims.size() + 1) {
    // Strides are not mutually exclusive. This could still be a
    // pipelinable access pattern, but we don't handle it for now.
    return std::nullopt;
  }

  // Verify that strides are ordered as expected.
  int64_t expected_stride = 1;
  for (auto [stride, dim] : strides_to_dims) {
    if (stride != expected_stride) {
      return std::nullopt;
    }
    Interval bound = indexing_map.GetDimensionBound(dim);
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

struct PipelinedLoadInfo {
  mlir::tensor::ExtractOp load_op;
  ApplyIndexingOp apply_indexing;
  IndexingMap base_indexing_map;

  int transaction_size;
  int64_t outer_loop_dim;

  Value thread_idx_value;
  int64_t thread_id_dim;
  int64_t thread_id_stride;
  llvm::SmallVector<int64_t> loop_var_dims;
  llvm::SmallVector<int64_t> loop_strides;
};

std::optional<PipelinedLoadInfo> GetLoadInfo(
    mlir::tensor::ExtractOp load,
    scf::ForOp outer_loop,
    llvm::SmallVector<scf::ForOp>& loop_nest) {
  if (load.getIndices().size() != 1){
    return std::nullopt;
  }
  auto apply_indexing = mlir::dyn_cast_or_null<ApplyIndexingOp>(
      load.getIndices().front().getDefiningOp());
  if (!apply_indexing) {
    return std::nullopt;
  }
  if (!IsConflictFree(load, outer_loop)) {
    return std::nullopt;
  }

  PipelinedLoadInfo info{load, apply_indexing, apply_indexing.getIndexingMap()};
  if (info.base_indexing_map.GetRangeVarsCount() != 0) {
    // The apply_indexing should have been canonicalized; there should
    // be no symbols left.
    return std::nullopt;
  }
  info.thread_id_dim = -1;
  info.outer_loop_dim = -1;

  auto strides = GetStrides(info.base_indexing_map.GetAffineMap().getResult(0));

  llvm::DenseMap<mlir::Value, mlir::AffineExpr> val_to_expr;
  for (int i = 0; i < apply_indexing.getNumOperands(); ++i) {
    auto expr = mlir::getAffineDimExpr(i, apply_indexing.getContext());
    Value operand = apply_indexing.getOperand(i);

    if (auto tid = mlir::dyn_cast_or_null<::mlir::gpu::ThreadIdOp>(operand.getDefiningOp())) {
      // TODO: verify this is x
      info.thread_id_dim = i;
      std::optional<int64_t> thread_id_stride = strides[expr];
      if (!thread_id_stride.has_value()) {
        return std::nullopt;
      }

      info.thread_id_stride = *thread_id_stride;
      info.thread_idx_value = operand;
    }

    if (operand == outer_loop.getInductionVar()) {
      info.outer_loop_dim = i;
    }

    if (!val_to_expr.try_emplace(operand, expr).second) {
      // TODO: check if this is actually possible. If it is, we need a canonicalization pattern for it.
      return std::nullopt;
    }
  }

  if (info.thread_id_dim == -1 || info.outer_loop_dim == -1) {
    return std::nullopt;
  }

  info.loop_var_dims.reserve(loop_nest.size());
  info.loop_strides.reserve(loop_nest.size());

  for (auto loop : loop_nest) {
    AffineExpr loop_var_expr = val_to_expr[loop.getInductionVar()];
    if (!loop_var_expr) {
      // e.g. broadcast. Just skip this for now.
      return std::nullopt;
    }

    info.loop_var_dims.push_back(
        mlir::cast<mlir::AffineDimExpr>(loop_var_expr).getPosition());
    auto stride = strides[loop_var_expr];
    if (!stride.has_value()) {
      return std::nullopt;
    }
    info.loop_strides.push_back(*stride);
  }

  std::optional<int64_t> transaction_size =
      GetPipelinedTransactionSize(
          info.base_indexing_map, info.thread_id_dim,
          info.thread_id_stride, info.loop_var_dims, info.loop_strides);
  if (!transaction_size.has_value()) {
    return std::nullopt;
  }

  info.transaction_size = *transaction_size;

  // Verify the non-loop dimensions are constant within outer_loop.
  for (int i = 0; i < apply_indexing.getNumOperands(); ++i) {
    if (i == info.thread_id_dim || i == info.outer_loop_dim) {
      continue;
    }

    if (std::find(info.loop_var_dims.begin(), info.loop_var_dims.end(), i) != 
        info.loop_var_dims.end()) {
      continue;
    }

    if (!apply_indexing.getOperand(i).getParentRegion()->isProperAncestor(
            &outer_loop.getBodyRegion())) {
      return std::nullopt;
    }
  }

  return info;
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
    llvm::SmallVector<scf::ForOp> loop_nest;  // inner to outer.
    while (loop) {
      loop_nest.push_back(loop);
      loop = mlir::dyn_cast_or_null<scf::ForOp>(loop->getParentOp());
    }

    // TODO: Hmm.
    loop = loop_nest.back();
    loop_nest.pop_back();

    auto trip_count = GetTripCount(loop);
    if (!trip_count || trip_count < 2) {
      return rewriter.notifyMatchFailure(op, "loop trip count unknown or too small");
    }

    auto maybe_info = GetLoadInfo(op, loop, loop_nest);
    if (!maybe_info) {
      return rewriter.notifyMatchFailure(op, "not a pipelineable load");
    }
    auto& info = *maybe_info;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(loop);
    auto zero = b.create<arith::ConstantIndexOp>(0);
    auto one = b.create<arith::ConstantIndexOp>(1);

    Value is_leader = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, info.thread_idx_value, zero);
    auto tx_ty = op.getTensor().getType().clone(llvm::SmallVector<int64_t>{info.transaction_size});
    auto empty_pipe_ty = gpu::SharedMemoryPipeType::get(
            b.getContext(),
            tx_ty,
            /*capacity=*/2, // TODO: always 2?
            /*level=*/0);
    Value pipe = b.create<gpu::AllocatePipeOp>(
        empty_pipe_ty,
        is_leader,
        128 /* TODO: get from domain */);

    // Prime the pipeline.
    llvm::SmallVector<mlir::Value> new_operands(
        info.apply_indexing.getOperands());
    new_operands[info.thread_id_dim] = zero;
    for (int64_t dim : info.loop_var_dims) {
      new_operands[dim] = zero;
    }
    for (auto idx : {zero, one}) {
      new_operands[info.outer_loop_dim] = idx;
      Value base_index = b.create<ApplyIndexingOp>(new_operands, info.base_indexing_map).getResult(0);
      llvm::SmallVector<OpFoldResult> offsets {base_index};
      llvm::SmallVector<OpFoldResult> sizes {b.getIndexAttr(info.transaction_size)};
      llvm::SmallVector<OpFoldResult> strides {b.getIndexAttr(1)};
      auto slice = b.create<mlir::tensor::ExtractSliceOp>(
          tx_ty, op.getTensor(), offsets, sizes, strides);
      pipe = b.create<gpu::EnqueueOp>(pipe, mlir::ValueRange(slice));
    }

    Value dequeued;
    (void) loop.replaceWithAdditionalYields(
        rewriter, pipe,
        /*replaceInitOperandUsesInLoop=*/false,
        [&](mlir::OpBuilder& yield_b, mlir::Location yield_loc,
            llvm::ArrayRef<mlir::BlockArgument> bbarg) -> llvm::SmallVector<Value> {
          mlir::ImplicitLocOpBuilder dequeue_builder(yield_loc, yield_b);
          dequeue_builder.setInsertionPointToStart(loop.getBody());
          Value orig_pipe = bbarg[0];
          auto value_and_pipe =
            dequeue_builder.create<gpu::DequeueOp>(orig_pipe).getResults();
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
                new_operands[info.outer_loop_dim] = induction_var_plus_two;
                Value base_index = then_b.create<ApplyIndexingOp>(then_loc, new_operands, info.base_indexing_map).getResult(0);
                llvm::SmallVector<OpFoldResult> offsets {base_index};
                llvm::SmallVector<OpFoldResult> sizes {then_b.getIndexAttr(info.transaction_size)};
                llvm::SmallVector<OpFoldResult> strides {then_b.getIndexAttr(1)};
                auto slice = then_b.create<mlir::tensor::ExtractSliceOp>(
                    then_loc, tx_ty, op.getTensor(), offsets, sizes, strides);
                Value new_pipe =
                    then_b.create<gpu::EnqueueOp>(then_loc, pipe, mlir::ValueRange(slice));
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
    
    llvm::SmallVector<Value> load_operands(
        info.base_indexing_map.GetDimVarsCount(), zero);
    load_operands[info.thread_id_dim] = info.thread_idx_value;

    AffineExpr load_offset =
        mlir::getAffineDimExpr(info.thread_id_dim, op.getContext()) *
        info.thread_id_stride;
    for (int i = 0; i < loop_nest.size(); ++i) {
      int64_t stride = info.loop_strides[i];
      int64_t dim = info.loop_var_dims[i];
      AffineExpr loop_var_expr =
         mlir::getAffineDimExpr(dim, op.getContext());
      load_offset = load_offset + loop_var_expr * stride;
      load_operands[dim] = loop_nest[i].getInductionVar();
    }
    mlir::AffineMap load_affine_map = mlir::AffineMap::get(
        info.base_indexing_map.GetDimVarsCount(), 0, load_offset);
    IndexingMap load_indexing_map(load_affine_map, info.base_indexing_map.GetDimVars(), {}, {});

    Value new_load_index = rewriter.create<ApplyIndexingOp>(
        op.getLoc(), load_operands, load_indexing_map).getResult(0);
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(
        op, dequeued, new_load_index);
    llvm::errs()  << *pipe.getDefiningOp()->getParentRegion()->begin() << "\n";
    return llvm::success();
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
