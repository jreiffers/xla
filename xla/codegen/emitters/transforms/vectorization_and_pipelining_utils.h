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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_VECTORIZATION_AND_PIPELINING_UTILS_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_VECTORIZATION_AND_PIPELINING_UTILS_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

namespace xla {
namespace emitters {

// For all dimensions and symbols in the expression, attempts to find the
// stride. If no stride can be determined, the expression will map to nullopt.
// Dimensions and symbols not present in the expression will also be absent
// from the result, even if a dimension or symbol with a higher index exists.
llvm::DenseMap<mlir::AffineExpr,
               std::optional<int64_t>> GetStrides(mlir::AffineExpr expr);

// Verifies that the given load is conflict-free with respect to the loop, i.e.
// the source tensor is constant within it.
bool IsConflictFree(mlir::tensor::ExtractOp load, mlir::scf::ForOp loop);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_VECTORIZATION_AND_PIPELINING_UTILS_H_
