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

std::optional<DynamicMemcpyThunk::MemcpyDescriptor> GetDynamicMemcpyDescriptor(
    const HloFusionAnalysis& analysis, const HloFusionInstruction* fusion,
    const IrEmitterCallStack& call_stack) {
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
    VLOG(5) << "Slice is not contiguous.";
    return std::nullopt;
  }

  std::optional<absl::InlinedVector<int64_t, 4>> strides =
      ShapeUtil::ByteStrides(slice.operand(0)->shape());
  if (!strides) {
    VLOG(5) << "Failed to get byte strides.";
    return std::nullopt;
  }

  VLOG(5) << "Preconditions passed, trying to build a memcpy descriptor.";
  DynamicMemcpyThunk::MemcpyDescriptor descriptor;
  for (int i = 0; i < slice.operand_count() - 1; ++i) {
    auto* operand = slice.operand(i + 1);
    // If this dimension's offset is always clamped to 0, we can skip it.
    if (slice.dynamic_slice_sizes()[i] ==
        slice.operand(0)->shape().dimensions(i)) {
      VLOG(5) << "Offset for dimension " << i << " is clamped to 0.";
      continue;
    }

    if (operand->opcode() == HloOpcode::kConstant) {
      std::optional<int64_t> value =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (!value) {
        return std::nullopt;
      }

      VLOG(5) << "Offset for dimension " << i << " is constant: " << *value
              << ".";
      descriptor.src_byte_static_offset += *value * (*strides)[i];
      continue;
    }

    auto functional_dependency = ResolveFunctionalDependencyOnInductionVariable(
        call_stack.instructions(), operand);
    if (!functional_dependency) {
      VLOG(5) << "Offset for dimension " << i << " is not statically known.";
      return std::nullopt;
    }

    // The while loop must actually be a for loop.
    auto loop_config = functional_dependency->loop
                           ->backend_config<xla::WhileLoopBackendConfig>();
    if (!loop_config.ok() || !loop_config->has_known_init_step() ||
        !loop_config->has_known_trip_count()) {
      VLOG(5) << "Offset for dimension " << i
              << " depends on loop with unknown behavior.";
      return std::nullopt;
    }

    VLOG(5) << "Offset for dimension " << i << " is dynamic.";
    descriptor.src_dynamic_offsets.emplace_back() = {
        functional_dependency->loop, functional_dependency->induction_var,
        functional_dependency->derived_value, (*strides)[i]};
  }

  return descriptor;
}

}  // namespace

std::optional<std::unique_ptr<FusionInterface>> HloFusionInfo::GetCopyFusion()
    const {
  // This type of fusion is not yet supported with command buffers.
  if (instr_->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_enable_command_buffer()
          .empty()) {
    auto dynamic_memcpy =
        GetDynamicMemcpyDescriptor(analysis(), instr_, call_stack_);
    if (dynamic_memcpy) {
      VLOG(5) << "Creating a dynamic memcpy fusion.";
      return std::make_unique<DynamicMemcpyFusion>(
          analysis(), buffer_assignment_, std::move(*dynamic_memcpy));
    }
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
