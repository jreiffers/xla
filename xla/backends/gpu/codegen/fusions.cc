/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/fusions.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/backends/gpu/codegen/cudnn.h"
#include "xla/backends/gpu/codegen/custom.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"
#include "xla/backends/gpu/codegen/emitters/input_slices.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/emitters/reduction.h"
#include "xla/backends/gpu/codegen/emitters/scatter.h"
#include "xla/backends/gpu/codegen/emitters/transpose.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

bool IsDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis) {
  return absl::c_all_of(
      analysis.fusion_roots(), [](const HloInstructionAdaptor& root) {
        return root.opcode() == HloOpcode::kDynamicUpdateSlice ||
               (root.opcode() == HloOpcode::kBitcast &&
                root.GetOperand(0).opcode() == HloOpcode::kDynamicUpdateSlice);
      });
}

// Will include the instruction itself.
// TODO(jreiffers): there's probably something in
// xla/hlo/analysis/hlo_reachability.h that we can use.
void GetDependencies(const HloInstruction* instruction,
                     absl::flat_hash_set<const HloInstruction*>& result) {
  if (result.insert(instruction).second) {
    for (auto* operand : instruction->operands()) {
      GetDependencies(operand, result);
    }
  }
}

// TODO(jreiffers): give this a better name.
struct WhileLoopSpec {
  const HloInstruction* loop;
  const HloInstruction* induction_var;
  const HloInstruction* slice_arg;
};

// TODO(jreiffers): Move this to copy.cc?
std::optional<WhileLoopSpec> GetDefiningWhileLoop(
    const IrEmitterScope& scope, const HloFusionInstruction* fusion,
    const HloInstruction* parameter) {
  if (scope.instructions().empty()) {
    return std::nullopt;
  }

  VLOG(5) << "Looking for defining while loop of " << parameter->name()
          << " in " << fusion->name();

  // Get out of the fusion.
  if (parameter->opcode() != HloOpcode::kParameter) {
    return std::nullopt;
  }

  auto* argument = fusion->operand(parameter->parameter_number());
  // Get out of the async fusion if it is one.
  if (argument->opcode() == HloOpcode::kParameter &&
      scope.instructions().back()->opcode() == HloOpcode::kAsyncStart) {
    argument =
        scope.instructions().back()->operand(argument->parameter_number());
  }

  // Find all the dependencies of the argument.
  absl::flat_hash_set<const HloInstruction*> deps;
  GetDependencies(argument, deps);

  // Find a unique parameter and a gte.
  const HloInstruction* unique_param = nullptr;
  const HloInstruction* unique_gte = nullptr;

  for (const auto* dep : deps) {
    if (dep->opcode() == HloOpcode::kParameter) {
      if (unique_param || !dep->shape().IsTuple()) {
        VLOG(5) << "Found wrong parameters.";
        return std::nullopt;
      }
      unique_param = dep;
    }

    if (dep->opcode() == HloOpcode::kGetTupleElement) {
      if (unique_gte) {
        VLOG(5) << "Found non-unique GTEs.";
        return std::nullopt;
      }
      unique_gte = dep;
    }
  }

  if (!unique_param || !unique_gte || unique_gte->operand(0) != unique_param) {
    VLOG(5) << "Did not find a parameter or GTE or they don't match.";
    return std::nullopt;
  }

  auto while_instr_it = std::find_if(
      scope.instructions().rbegin(), scope.instructions().rend(),
      [&](const HloInstruction* instr) {
        if (instr->opcode() != HloOpcode::kWhile) return false;

        // Verify that this GTE is the induction variable of the loop.
        if (unique_param != instr->while_body()->parameter_instruction(0)) {
          return false;
        }
        if (unique_gte->tuple_index() != GetLoopInductionVarTupleIdx(instr)) {
          return false;
        }
        return true;
      });
  if (while_instr_it == scope.instructions().rend()) {
    return std::nullopt;
  }
  return WhileLoopSpec{*while_instr_it, unique_gte, argument};
}

std::optional<DynamicMemcpyThunk::MemcpyDescriptor> GetDynamicMemcpyDescriptor(
    const HloFusionAnalysis& analysis, const HloFusionInstruction* fusion,
    const IrEmitterScope& scope) {
  VLOG(5) << "Looking for a memcpy in " << fusion->name() << ".";
  if (analysis.fusion_roots().size() != 1) {
    return std::nullopt;
  }

  auto root = analysis.fusion_roots().front();
  if (root.opcode() != HloOpcode::kDynamicSlice) {
    return std::nullopt;
  }

  const auto& slice = root.instruction();
  if (slice.operand(0)->opcode() != HloOpcode::kParameter) {
    return std::nullopt;
  }

  // Only contiguous slices can be represented by a memcpy.
  if (!IsContiguousSlice(slice)) {
    return std::nullopt;
  }

  std::optional<absl::InlinedVector<int64_t, 4>> strides =
      ShapeUtil::ByteStrides(slice.operand(0)->shape());
  if (!strides) {
    return std::nullopt;
  }

  DynamicMemcpyThunk::MemcpyDescriptor descriptor;
  for (int i = 0; i < slice.operand_count() - 1; ++i) {
    auto* operand = slice.operand(i + 1);
    // If this dimension's offset is always clamped to 0, we can skip it.
    if (slice.dynamic_slice_sizes()[i] ==
        slice.operand(0)->shape().dimensions(i)) {
      continue;
    }

    if (operand->opcode() == HloOpcode::kConstant) {
      std::optional<int64_t> value =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (!value) {
        return std::nullopt;
      }

      descriptor.src_byte_static_offset += *value * (*strides)[i];
      continue;
    }

    auto loop = GetDefiningWhileLoop(scope, fusion, operand);
    if (loop) {
      descriptor.src_dynamic_offsets.emplace_back() = {
          loop->loop, loop->induction_var, loop->slice_arg, (*strides)[i]};
      continue;
    }

    return std::nullopt;
  }

  return descriptor;
}

}  // namespace

std::optional<std::unique_ptr<FusionInterface>> HloFusionInfo::GetCopyFusion()
    const {
  auto dynamic_memcpy = GetDynamicMemcpyDescriptor(analysis(), instr_, scope_);
  if (dynamic_memcpy) {
    return std::make_unique<DynamicMemcpyFusion>(analysis(), buffer_assignment_,
                                                 std::move(*dynamic_memcpy));
  }

  for (const HloInstructionAdaptor& root_adaptor : analysis().fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }
  }

  return std::make_unique<MemcpyFusion>(analysis(), buffer_assignment_);
}

bool HloFusionInfo::CanEmitDynamicUpdateSliceInPlace() const {
  auto ret = CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
      analysis().fusion(),
      [this](const HloInstruction* instruction, const ShapeIndex& index) {
        return GetAllocationSlice(*buffer_assignment_, instruction, index);
      },
      instr_);
  return ret.ok() && *ret;
}

std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo& fusion_info) {
  const auto& analysis = fusion_info.analysis();
  const FusionBackendConfig& backend_config = analysis.fusion_backend_config();

  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion: {
      const absl::string_view& config_name =
          backend_config.custom_fusion_config().name();
      if (config_name ==
              kDynamicSliceFusionWithStaticAddressComputationConfigName ||
          config_name ==
              kDynamicSliceFusionWithDynamicAddressComputationConfigName) {
        const HloFusionInfo* hlo_fusion_info =
            dynamic_cast<const HloFusionInfo*>(&fusion_info);
        return std::make_unique<DynamicSliceFusion>(
            analysis, hlo_fusion_info->GetCallGraph());
      }
      return std::make_unique<CustomFusion>();
    }
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
      return std::make_unique<InputSlicesFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      if (IsDynamicUpdateSliceFusion(analysis) &&
          fusion_info.CanEmitDynamicUpdateSliceInPlace()) {
        return std::make_unique<InPlaceDynamicUpdateSliceFusion>(analysis);
      }
      if (auto copy_fusion = fusion_info.GetCopyFusion()) {
        return *std::move(copy_fusion);
      }
      return std::make_unique<LoopFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return CreateReductionFusion(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kScatter: {
      return CreateScatterFusion(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kTranspose: {
      return std::make_unique<TransposeFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate: {
      return std::make_unique<ConcatenateFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
      return std::make_unique<TritonFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return std::make_unique<CuDnnFusion>(analysis);
  }
}

}  // namespace gpu
}  // namespace xla
