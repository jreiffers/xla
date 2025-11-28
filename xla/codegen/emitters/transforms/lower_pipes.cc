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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/status.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"            // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_LOWERPIPESPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

using llvm::ArrayRef;
using llvm::dyn_cast_or_null;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpResult;
using mlir::OpRewritePattern;
using mlir::SmallVector;
using mlir::success;
using mlir::Type;
using mlir::TypedValue;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace ml = ::mlir::LLVM;
namespace vector = ::mlir::vector;

// Convenience wrapper for creating an InlineAsmOp with the common constant
// arguments used throughout this file. Use this when the call should have
// has_side_effects=true and is_align_stack=false (the common case). Calls
// that need a different is_align_stack value should continue to construct
// the InlineAsmOp directly.
static ml::InlineAsmOp CreateInlineAsm(OpBuilder& b, Location loc,
                                       TypeRange resultTypes,
                                       ValueRange operands,
                                       const std::string& asm_string,
                                       const std::string& constraints) {
  auto asm_dialect =
      ml::AsmDialectAttr::get(b.getContext(), ml::AsmDialect::AD_ATT);
  return b.create<ml::InlineAsmOp>(
      loc, resultTypes, operands, asm_string, constraints,
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, ml::TailCallKind::None, asm_dialect,
      /*operand_attrs=*/mlir::ArrayAttr());
}

ml::GlobalOp CreateGlobalOp(mlir::Attribute value,
                            const std::string& name_prefix,
                            mlir::ShapedType shaped_ty, mlir::ModuleOp module,
                            bool is_constant, int addr_space,
                            mlir::ImplicitLocOpBuilder& b) {
  if (auto elements = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(value)) {
    value = elements.reshape(mlir::RankedTensorType::get(
        {elements.getNumElements()}, elements.getElementType()));
  }

  Type element_type = shaped_ty.getElementType();
  int64_t num_elements = shaped_ty.getNumElements();
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  if (element_type.isIntOrFloat() &&
      element_type.getIntOrFloatBitWidth() == 4) {
    num_elements = CeilOfRatio<int64_t>(num_elements, 2);
    llvm_element_type = b.getI8Type();
    if (value) {
      auto unpacked_data =
          mlir::cast<mlir::DenseElementsAttr>(value).getRawData();
      std::vector<char> packed_data(num_elements);
      absl::Span<char> packed_data_span =
          absl::MakeSpan(packed_data.data(), packed_data.size());
      PackIntN(4, unpacked_data, packed_data_span);
      value = mlir::DenseElementsAttr::getFromRawBuffer(
          mlir::RankedTensorType::get({num_elements}, llvm_element_type),
          packed_data);
    }
  }
  auto array_ty = ml::LLVMArrayType::get(llvm_element_type, num_elements);
  std::string name;
  int index = 0;
  do {
    name = absl::StrCat(name_prefix, index);
    ++index;
  } while (module.lookupSymbol(name));
  b.setInsertionPointToStart(module.getBody());
  return b.create<ml::GlobalOp>(array_ty, is_constant,
                                /*linkage=*/ml::Linkage::Private, name, value,
                                /*alignment=*/0, addr_space);
}

mlir::Value CreateSharedMemory(mlir::Operation* op,
                               mlir::ShapedType shared_memory_shape,
                               mlir::PatternRewriter& rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  constexpr int kGPUSharedMemoryAddrSpace = 3;
  mlir::ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  auto global =
      CreateGlobalOp(mlir::Attribute{}, "shared_", shared_memory_shape, module,
                     /*is_constant=*/false, kGPUSharedMemoryAddrSpace, b);

  rewriter.setInsertionPoint(op);
  auto addr = rewriter.create<ml::AddressOfOp>(op->getLoc(), global);
  return rewriter
      .create<ml::AddrSpaceCastOp>(
          op->getLoc(), ml::LLVMPointerType::get(op->getContext()), addr)
      .getResult();
}

Value GetLinearIndex(ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  CHECK_LE(indices.size(), 1) << "Only 0D and 1D tensors are supported";
  auto index = indices.empty() ? b.create<mlir::arith::ConstantIndexOp>(0)
                               : indices.front();
  auto index_ty = b.getIntegerType(
      mlir::DataLayout::closest(b.getInsertionBlock()->getParentOp())
          .getTypeSizeInBits(index.getType()));
  return b.create<mlir::arith::IndexCastUIOp>(index_ty, index);
}

std::optional<int> GetSubByteBitWidth(Type element_type) {
  if (element_type.isIntOrFloat()) {
    int bit_width = element_type.getIntOrFloatBitWidth();
    if (bit_width == 4 || bit_width == 2) {
      return bit_width;
    }
  }
  return std::nullopt;
}

int SubByteIndexingBits(int bit_width) {
  CHECK_LT(bit_width, 8) << "Passed width is not a sub-byte";
  return absl::bit_width<unsigned int>(8 / bit_width) - 1;
}

std::tuple<Value, Value> GetSubByteIndex(Value linear_index, int bit_width,
                                         mlir::ImplicitLocOpBuilder& b) {
  CHECK_LT(bit_width, 8) << "Passed width is not a sub-byte";
  int sub_byte_indexing_bits = SubByteIndexingBits(bit_width);
  Value sub_byte_index = b.create<mlir::arith::TruncIOp>(
      b.getI8Type(), b.create<mlir::arith::AndIOp>(
                         linear_index, b.create<mlir::arith::ConstantIntOp>(
                                           linear_index.getType(),
                                           (1 << sub_byte_indexing_bits) - 1)));
  Value sub_byte_shift = b.create<mlir::arith::MulIOp>(
      b.create<mlir::arith::ConstantIntOp>(b.getI8Type(), bit_width),
      sub_byte_index);
  Value byte_shift = b.create<mlir::arith::ConstantIntOp>(
      linear_index.getType(), sub_byte_indexing_bits);
  Value i8_index = b.create<mlir::arith::ShRUIOp>(linear_index, byte_shift);
  return {i8_index, sub_byte_shift};
}

ml::GEPOp CreateGep(Value tensor_ptr, mlir::RankedTensorType tensor_type,
                    Value linear_index, mlir::ImplicitLocOpBuilder& b) {
  auto element_type = tensor_type.getElementType();
  int64_t num_elements = tensor_type.getNumElements();
  std::optional<int> sub_byte_width = GetSubByteBitWidth(element_type);
  if (sub_byte_width) {
    element_type = b.getI8Type();
    num_elements = CeilOfRatio<int64_t>(num_elements, 8 / *sub_byte_width);
  }
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  auto array_type =
      b.getType<ml::LLVMArrayType>(llvm_element_type, num_elements);
  auto ptr = ml::LLVMPointerType::get(b.getContext());
  if (linear_index.getType().isIndex()) {
    linear_index = GetLinearIndex(linear_index, b);
  }
  auto gep = b.create<ml::GEPOp>(
      ptr, array_type, tensor_ptr,
      llvm::SmallVector<mlir::LLVM::GEPArg>{0, linear_index});
  gep.setNoWrapFlags(mlir::LLVM::GEPNoWrapFlags::inbounds);
  return gep;
}

ml::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                    Value linear_index, mlir::ImplicitLocOpBuilder& b) {
  mlir::RankedTensorType tensor_type = tensor.getType();
  auto ptr = ml::LLVMPointerType::get(b.getContext());
  auto tensor_ptr =
      b.create<UnrealizedConversionCastOp>(ptr, tensor).getResult(0);
  return CreateGep(tensor_ptr, tensor_type, linear_index, b);
}

ml::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                    ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  return CreateGep(tensor, GetLinearIndex(indices, b), b);
}

mlir::RankedTensorType GetMbarsTensorType(gpu::SharedMemoryPipeType ty,
                                          mlir::OpBuilder& b) {
  return mlir::RankedTensorType::get({ty.getCapacity()}, b.getIntegerType(64));
}

struct ConvertAllocatePipe
    : public mlir::OpConversionPattern<gpu::AllocatePipeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::AllocatePipeOp op, OneToNOpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto pipe_ty = op.getResult().getType();

    SmallVector<Type> pipe_types;
    if (failed(getTypeConverter()->convertTypes(pipe_ty, pipe_types)))
      return failure();

    mlir::ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto read_idx = b.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto mbars = b.create<gpu::AllocateSharedOp>(pipe_types[0]);

    auto mbars_init = b.create<gpu::InitMembarsOp>(mbars, op.getThreadCount(),
                                                   op.getLeader());
    SmallVector<Value> result{mbars_init, read_idx, op.getLeader()};
    for (int i = 3; i < pipe_types.size(); ++i) {
      auto buf = b.create<gpu::AllocateSharedOp>(pipe_types[i]);
      result.push_back(buf);
    }

    rewriter.replaceOpWithMultiple(op, {result});
    return success();
  }
};

struct ConvertEnqueue : public mlir::OpConversionPattern<gpu::EnqueueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::EnqueueOp op, OneToNOpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto in_pipe = adaptor.getPipe();
    Value membars = in_pipe[0];
    Value read_index = in_pipe[1];
    Value leader = in_pipe[2];
    ValueRange buffers = in_pipe.drop_front(3);

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value level =
        b.create<arith::ConstantIndexOp>(op.getPipe().getType().getLevel());
    Value capacity =
        b.create<arith::ConstantIndexOp>(op.getPipe().getType().getCapacity());

    // Compute the write index from the read index and the level.
    Value write_index = b.create<arith::RemUIOp>(
        b.create<arith::AddIOp>(read_index, level), capacity);
    auto copy_start = b.create<gpu::AsyncCopyStartOp>(
        mlir::TypeRange(buffers), membars.getType(), op.getSources(), buffers,
        membars, write_index, leader);
  
    SmallVector<Value> result{copy_start.getOutMembars(), read_index, leader};
    result.append(copy_start.getOutBuffers().begin(),
                  copy_start.getOutBuffers().end());

    rewriter.replaceOpWithMultiple(op, {result});
    return success();
  }
};

struct ConvertDequeue : public mlir::OpConversionPattern<gpu::DequeueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::DequeueOp op, OneToNOpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto in_pipe = adaptor.getPipe();
    Value membars = in_pipe[0];
    Value read_index = in_pipe[1];
    Value leader = in_pipe[2];
    ValueRange buffers = in_pipe.drop_front(3);

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value one = b.create<arith::ConstantIndexOp>(1);
    Value capacity =
        b.create<arith::ConstantIndexOp>(op.getPipe().getType().getCapacity());
    Value next_read_index = b.create<arith::RemUIOp>(
        b.create<arith::AddIOp>(read_index, one), capacity);
    auto copy_wait = b.create<gpu::AsyncCopyWaitOp>(
        mlir::TypeRange(op.getTensors()), membars.getType(), buffers, membars,
        read_index, leader);

    SmallVector<SmallVector<Value>> results;
    for (auto out : copy_wait.getOutElements()) {
      results.push_back({out});
    }
    SmallVector<Value>& result_pipe = results.emplace_back();
    result_pipe = {copy_wait.getOutMembars(), next_read_index, leader};
    result_pipe.append(buffers.begin(), buffers.end());

    rewriter.replaceOpWithMultiple(op, results);
    return success();
  }
};

struct ConvertEnqueueUndef
    : public mlir::OpConversionPattern<gpu::EnqueueUndefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gpu::EnqueueUndefOp op, OneToNOpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithMultiple(
        op, SmallVector<SmallVector<Value>>{adaptor.getPipe()});
    return success();
  }
};

}  // namespace

class LowerPipesPass : public impl::LowerPipesPassBase<LowerPipesPass> {
 public:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();

    // Build a TypeConverter that only converts the SharedMemoryPipeType to an
    // LLVM pointer. All other types are passed through unchanged. We add
    // source/target materializations that use `unrealized_conversion_cast`
    // to bridge between the original pipe type and the converted pointer.
    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    // Convert the SharedMemoryPipeType into the 4-tuple of types that make
    // up the unpacked pipe representation: (buffer tensor, mbars tensor,
    // read index, is_leader i1). This is a 1->N conversion.
    typeConverter.addConversion([](gpu::SharedMemoryPipeType pipeTy,
                                   llvm::SmallVectorImpl<Type>& out)
                                    -> std::optional<mlir::LogicalResult> {
      MLIRContext* ctx = pipeTy.getContext();
      Type mbars_ty = mlir::RankedTensorType::get(
          {pipeTy.getCapacity()}, mlir::IntegerType::get(ctx, 64));
      out.push_back(mbars_ty);
      out.push_back(mlir::IndexType::get(ctx));
      out.push_back(mlir::IntegerType::get(ctx, 1));

      for (auto elem_ty : pipeTy.getElementTypes()) {
        SmallVector<int64_t> buf_shape;
        buf_shape.push_back(pipeTy.getCapacity());
        buf_shape.append(elem_ty.getShape().begin(), elem_ty.getShape().end());
        if (buf_shape.empty()) return std::nullopt;
        Type buffer_ty = elem_ty.clone(buf_shape);
        out.push_back(buffer_ty);
      }
      return mlir::success();
    });

    typeConverter.addSourceMaterialization(
        [](mlir::OpBuilder& builder, mlir::Type resultType,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
        });

    typeConverter.addTargetMaterialization(
        [](mlir::OpBuilder& builder, mlir::TypeRange resultTypes,
           mlir::ValueRange inputs,
           mlir::Location loc) -> SmallVector<mlir::Value> {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultTypes, inputs)
              .getResults();
        });

    // Build conversion target and patterns that replace the four pipe ops
    // with the converted representation (1->N conversion on the pipe type).
    mlir::ConversionTarget target(*ctx);
    // Allow unknown ops to remain legal; only mark our pipe ops illegal so
    // that applyPartialConversion will rewrite them.
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<gpu::AllocatePipeOp, gpu::EnqueueOp,
                        gpu::EnqueueUndefOp, gpu::DequeueOp>();

    mlir::RewritePatternSet patterns(ctx);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    patterns.add<ConvertAllocatePipe, ConvertEnqueue, ConvertEnqueueUndef,
                 ConvertDequeue>(typeConverter, ctx);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<::mlir::Pass> CreateLowerPipesPass() {
  return std::make_unique<LowerPipesPass>();
}

}  // namespace emitters
}  // namespace xla
