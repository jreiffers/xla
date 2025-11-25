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

#include "xla/codegen/emitters/transforms/vectorization_and_pipelining_utils.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/AffineExpr.h"

namespace xla {
namespace emitters {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;

template <typename Fn>
void VisitSummands(AffineExpr expr, const Fn& visit) {
  if (expr.getKind() == AffineExprKind::Add) {
    auto binop = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
    VisitSummands(binop.getLHS(), visit);
    VisitSummands(binop.getRHS(), visit);
  } else {
    visit(expr);
  }
}

std::optional<std::pair<AffineExpr, int64_t>> GetConstantMultiplier(
    AffineExpr expr) {
  if (expr.getKind() == AffineExprKind::Mul) {
    auto mul_expr = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = mul_expr.getLHS();
    auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(mul_expr.getRHS());
    if (rhs && (lhs.getKind() == AffineExprKind::DimId ||
                lhs.getKind() == AffineExprKind::SymbolId)) {
      return std::make_pair(lhs, rhs.getValue());
    }
  }

  if (expr.getKind() == AffineExprKind::DimId ||
      expr.getKind() == AffineExprKind::SymbolId) {
    // Multiplication by constant 1.
    return std::make_pair(expr, 1);
  }

  return std::nullopt;
}

}

llvm::DenseMap<AffineExpr,
               std::optional<int64_t>> GetStrides(AffineExpr expr) {
  llvm::DenseMap<AffineExpr, std::optional<int64_t>> result;
  VisitSummands(expr, [&](AffineExpr summand) {
    // If `expr` is a multiplication by a constant, we can determine the stride
    // for the LHS if it is a symbol or dimension. Otherwise, we set the strides
    // of all dimensions and symbols occurring in `expr` to nullopt (indicating
    // unknown).
    // We assume the expression is simplified, so the constant will always be
    // on the RHS.
    if (auto constant_multiplier = GetConstantMultiplier(summand)) {
      auto [leaf, stride] = *constant_multiplier;
      auto it_and_inserted = result.try_emplace(leaf, stride);
      if (it_and_inserted.second) {
        return;
      } else if (it_and_inserted.first->second.has_value()) {
        // There was already an entry with a known stride, so we can sum them.
        // Technically this can't happen, because we assume the expression is
        // simplified.
        it_and_inserted.first->second.value() += stride;
        return;
      }
    }

    // If we fell through, we failed to determine the stride. Set all dimensions
    // and symbols in `summand` to nullopt.
    summand.walk([&](AffineExpr leaf) {
      if (mlir::isa<mlir::AffineSymbolExpr, mlir::AffineDimExpr>(leaf)) {
        result[leaf] = std::nullopt;
      }
    });
  });
  return result;
}

bool IsConflictFree(mlir::tensor::ExtractOp load, mlir::scf::ForOp loop) {
  return load.getTensor().getParentRegion()->isProperAncestor(
      &loop.getBodyRegion());
}

}  // namespace emitters
}  // namespace xla