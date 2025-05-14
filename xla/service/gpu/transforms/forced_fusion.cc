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

#include "xla/service/gpu/transforms/forced_fusion.h"

namespace xla {
namespace gpu {

namespace {

bool IsMustFuseCall(const HloInstruction* caller) {
  const auto& attrs = caller->frontend_attributes().map();
  auto must_fuse_it = attrs.find("MUST_FUSE");
  return must_fuse_it != attrs.end() && must_fuse_it->second == "true";
}

}  // namespace

absl::StatusOr<bool> ForcedFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto computations = module->MakeComputationPostOrder(execution_threads);

  bool changed = false;
  for (auto* computation : computations) {
    auto callers = computation->caller_instructions(HloOpcode::kCall);
    if (callers.size() > 1) {
      for (auto* caller : callers) {
        TF_RET_CHECK(!IsMustFuseCall(caller))
            << "Callers with MUST_FUSE should be the only user of their "
               "computation.";
      }
    }

    if (callers.size() != 1 || !IsMustFuseCall(callers.front())) {
      continue;
    }

    auto* caller = callers.front();
    // TODO(jreiffers): Handle edge cases regarding location of caller? Maybe
    // just check we're in the entry computation or some expected control flow.

    // TODO(jreiffers): Sharding?
    // TODO(jreiffers): Control dependencies?
    caller->ClearCalledComputations();
    TF_RETURN_IF_ERROR(caller->parent()->ReplaceWithNewInstruction(
        caller, HloInstruction::CreateFusion(
                    caller->shape(),
                    // TODO(jreiffers): Get rid of fusion kind in XLA:GPU.
                    // It doesn't do anything useful.
                    HloInstruction::FusionKind::kLoop, caller->operands(),
                    computation)));
    changed = true;
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla