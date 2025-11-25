// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-pipes -canonicalize -xla-flatten-tensors -xla-lower-tensors --mlir-print-ir-before-all --mlir-print-ir-after-all  \


// xRUN: | FileCheck %s --dump-input=always

#map = #xla.indexing_map<"(d0)[s0] -> (d0, s0),"
                             "domain:"
                             "d0 in [0, 128],"
                             "s0 in [0, 128],"
                        >

func.func @pipes(%is_lead: i1, %num_threads: i32, %a: tensor<1024xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %pipe = xla_gpu.allocate_pipe(%is_lead, 128) : !xla_gpu.shmem_pipe<4*tensor<128xf32>, 0>
  %slice0 = tensor.extract_slice %a[0][128][1]
      : tensor<1024xf32> to tensor<128xf32>
  %slice1 = tensor.extract_slice %a[128][128][1]
      : tensor<1024xf32> to tensor<128xf32>
  %enq1 = xla_gpu.enqueue %slice0 into %pipe : tensor<128xf32> into !xla_gpu.shmem_pipe<4*tensor<128xf32>, 0>
  %enq2 = xla_gpu.enqueue %slice1 into %enq1 : tensor<128xf32> into !xla_gpu.shmem_pipe<4*tensor<128xf32>, 1>
  %val1, %deq1 = xla_gpu.dequeue %enq2 : !xla_gpu.shmem_pipe<4*tensor<128xf32>, 2>
  %val2, %deq2 = xla_gpu.dequeue %deq1 : !xla_gpu.shmem_pipe<4*tensor<128xf32>, 1>
  "dummy.consume"(%val1, %val2) : (tensor<128xf32>, tensor<128xf32>) -> ()
  return %val1, %val2 : tensor<128xf32>, tensor<128xf32>
}

// Output:
// func.func @pipes(%arg0: i1, %arg1: i32, %arg2: !llvm.ptr) {
//   %0 = llvm.mlir.addressof @shared_3 : !llvm.ptr<3>
//   %1 = llvm.mlir.addressof @shared_2 : !llvm.ptr<3>
//   %c128_i32 = arith.constant 128 : i32
//   %c512_i64 = arith.constant 512 : i64
//   %c0_i32 = arith.constant 0 : i32
//   %true = arith.constant true
//   %2 = llvm.addrspacecast %0 : !llvm.ptr<3> to !llvm.ptr
//   %3 = llvm.addrspacecast %1 : !llvm.ptr<3> to !llvm.ptr
//   scf.if %arg0 {
//     %18 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %18, %c128_i32 : !llvm.ptr, i32
//     %19 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %19, %c128_i32 : !llvm.ptr, i32
//     %20 = llvm.getelementptr inbounds %3[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %20, %c128_i32 : !llvm.ptr, i32
//     %21 = llvm.getelementptr inbounds %3[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %21, %c128_i32 : !llvm.ptr, i32
//   }
//   gpu.barrier
//   %4 = llvm.getelementptr inbounds %arg2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x f32>
//   %5 = llvm.getelementptr inbounds %arg2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x f32>
//   %6 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %7 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   scf.if %arg0 {
//     llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], 512, [$2];", "l,l,r,l" %6, %4, %7 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
//   }
//   %8 = llvm.getelementptr inbounds %2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %9 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   scf.if %arg0 {
//     llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], 512, [$2];", "l,l,r,l" %8, %5, %9 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
//   }
//   %10 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %11 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   %12 = scf.if %arg0 -> (i64) {
//     %18 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r" %11, %c512_i64 : (!llvm.ptr, i64) -> i64
//     scf.yield %18 : i64
//   } else {
//     %18 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.shared.b64 $0, [$1];", "=l,l" %11 : (!llvm.ptr) -> i64
//     scf.yield %18 : i64
//   }
//   %13 = scf.while (%arg3 = %true) : (i1) -> i1 {
//     %18 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.try_wait.shared::cta.b64 $0, [$1], $2;", "=l,l,l" %11, %12 : (!llvm.ptr, i64) -> i32
//     %19 = arith.cmpi ne, %18, %c0_i32 : i32
//     scf.condition(%19) %19 : i1
//   } do {
//   ^bb0(%arg3: i1):
//     scf.yield %arg3 : i1
//   }
//   %14 = llvm.getelementptr inbounds %2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %15 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   %16 = scf.if %arg0 -> (i64) {
//     %18 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r" %15, %c512_i64 : (!llvm.ptr, i64) -> i64
//     scf.yield %18 : i64
//   } else {
//     %18 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.shared.b64 $0, [$1];", "=l,l" %15 : (!llvm.ptr) -> i64
//     scf.yield %18 : i64
//   }
//   %17 = scf.while (%arg3 = %true) : (i1) -> i1 {
//     %18 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.try_wait.shared::cta.b64 $0, [$1], $2;", "=l,l,l" %15, %16 : (!llvm.ptr, i64) -> i32
//     %19 = arith.cmpi ne, %18, %c0_i32 : i32
//     scf.condition(%19) %19 : i1
//   } do {
//   ^bb0(%arg3: i1):
//     scf.yield %arg3 : i1
//   }
//   "dummy.consume"(%10, %14) : (!llvm.ptr, !llvm.ptr) -> ()
//   return
// }

// Test: pipelined TMA.

func.func @pipe_loop2(%is_lead: i1, %num_threads: i32, %a: tensor<10x128xf32>) -> (f32) {
  %pipe = xla_gpu.allocate_pipe(%is_lead, 128) : !xla_gpu.shmem_pipe<2*tensor<128xf32>, 0>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  // Prime the queue with two enqueues.
  %slice0 = tensor.extract_slice %a[0, 0][1,128][1,1]
      : tensor<10x128xf32> to tensor<128xf32>
  %slice1 = tensor.extract_slice %a[1, 0][1,128][1,1]
      : tensor<10x128xf32> to tensor<128xf32>
  %enq1 = xla_gpu.enqueue %slice0 into %pipe
      : tensor<128xf32> into !xla_gpu.shmem_pipe<2*tensor<128xf32>, 0>
  %enq2 = xla_gpu.enqueue %slice1 into %enq1
      : tensor<128xf32> into !xla_gpu.shmem_pipe<2*tensor<128xf32>, 1>

  %zero = arith.constant 0.0 : f32
  %_, %sum_res = scf.for %i = %c0 to %c10 step %c1 iter_args(%p = %enq2, %sum = %zero)
      -> (!xla_gpu.shmem_pipe<2*tensor<128xf32>, 2>, f32) {
    // Dequeue one element.
    %val, %p_after = xla_gpu.dequeue %p : !xla_gpu.shmem_pipe<2*tensor<128xf32>, 2>

    %v = tensor.extract %val[%c0] : tensor<128xf32>
    %sum_next = arith.addf %sum, %v : f32

    // Compute next enqueue or enqueue_undef for the last two iterations.
    %i_plus2 = arith.addi %i, %c2 : index
    %off = arith.muli %i_plus2, %c128 : index
    %is_tail = arith.cmpi sge, %i, %c8 : index
    %p_next = scf.if %is_tail -> !xla_gpu.shmem_pipe<2*tensor<128xf32>, 2> {
      %u = xla_gpu.enqueue_undef %p_after
          : !xla_gpu.shmem_pipe<2*tensor<128xf32>, 1>
      scf.yield %u : !xla_gpu.shmem_pipe<2*tensor<128xf32>, 2>
    } else {
      %next_slice = tensor.extract_slice %a[%off, 0][1,128][1,1]
          : tensor<10x128xf32> to tensor<128xf32>
      %p_e = xla_gpu.enqueue %next_slice into %p_after
          : tensor<128xf32> into !xla_gpu.shmem_pipe<2*tensor<128xf32>, 1>
      scf.yield %p_e : !xla_gpu.shmem_pipe<2*tensor<128xf32>, 2>
    }

    scf.yield %p_next, %sum_next
        : !xla_gpu.shmem_pipe<2*tensor<128xf32>, 2>, f32
  }

  return %sum_res : f32
}

// Output:
// func.func @pipe_loop2(%arg0: i1, %arg1: i32, %arg2: !llvm.ptr) -> f32 {
//   %0 = llvm.mlir.addressof @shared_1 : !llvm.ptr<3>
//   %cst = arith.constant 0.000000e+00 : f32
//   %c128 = arith.constant 128 : index
//   %c10 = arith.constant 10 : index
//   %c8 = arith.constant 8 : index
//   %c2 = arith.constant 2 : index
//   %c1 = arith.constant 1 : index
//   %c0 = arith.constant 0 : index
//   %1 = llvm.mlir.addressof @shared_0 : !llvm.ptr<3>
//   %c128_i32 = arith.constant 128 : i32
//   %c512_i64 = arith.constant 512 : i64
//   %c0_i32 = arith.constant 0 : i32
//   %true = arith.constant true
//   %2 = llvm.addrspacecast %0 : !llvm.ptr<3> to !llvm.ptr
//   %3 = llvm.addrspacecast %1 : !llvm.ptr<3> to !llvm.ptr
//   scf.if %arg0 {
//     %11 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
//     nvvm.mbarrier.init %11, %c128_i32 : !llvm.ptr, i32
//     %12 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
//     nvvm.mbarrier.init %12, %c128_i32 : !llvm.ptr, i32
//   }
//   gpu.barrier
//   %4 = llvm.getelementptr inbounds %arg2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1280 x f32>
//   %5 = llvm.getelementptr inbounds %arg2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1280 x f32>
//   %6 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x f32>
//   %7 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
//   scf.if %arg0 {
//     llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], 512, [$2];", "l,l,r,l" %6, %4, %7 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
//   }
//   %8 = llvm.getelementptr inbounds %2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x f32>
//   %9 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
//   scf.if %arg0 {
//     llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], 512, [$2];", "l,l,r,l" %8, %5, %9 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
//   }
//   %10:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %c0, %arg5 = %cst) -> (index, f32) {
//     %11 = arith.addi %arg4, %c1 : index
//     %12 = arith.remui %11, %c2 : index
//     %13 = arith.muli %arg4, %c128 : index
//     %14 = arith.index_castui %13 : index to i64
//     %15 = llvm.getelementptr inbounds %2[0, %14] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x f32>
//     %16 = arith.index_castui %arg4 : index to i64
//     %17 = llvm.getelementptr inbounds %3[0, %16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i64>
//     %18 = scf.if %arg0 -> (i64) {
//       %26 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r" %17, %c512_i64 : (!llvm.ptr, i64) -> i64
//       scf.yield %26 : i64
//     } else {
//       %26 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.shared.b64 $0, [$1];", "=l,l" %17 : (!llvm.ptr) -> i64
//       scf.yield %26 : i64
//     }
//     %19 = scf.while (%arg6 = %true) : (i1) -> i1 {
//       %26 = llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.try_wait.shared::cta.b64 $0, [$1], $2;", "=l,l,l" %17, %18 : (!llvm.ptr, i64) -> i32
//       %27 = arith.cmpi ne, %26, %c0_i32 : i32
//       scf.condition(%27) %27 : i1
//     } do {
//     ^bb0(%arg6: i1):
//       scf.yield %arg6 : i1
//     }
//     %20 = llvm.getelementptr inbounds %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x f32>
//     %21 = llvm.load %20 : !llvm.ptr -> f32
//     %22 = arith.addf %arg5, %21 : f32
//     %23 = arith.addi %arg3, %c2 : index
//     %24 = arith.muli %23, %c128 : index
//     %25 = arith.cmpi sge, %arg3, %c8 : index
//     scf.if %25 {
//     } else {
//       %26 = xla.apply_indexing #indexing_map(%24)
//       %27 = arith.index_castui %26 : index to i64
//       %28 = llvm.getelementptr inbounds %arg2[0, %27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1280 x f32>
//       %29 = arith.addi %12, %c1 : index
//       %30 = arith.remui %29, %c2 : index
//       %31 = arith.muli %30, %c128 : index
//       %32 = arith.index_castui %31 : index to i64
//       %33 = llvm.getelementptr inbounds %2[0, %32] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x f32>
//       %34 = arith.index_castui %30 : index to i64
//       %35 = llvm.getelementptr inbounds %3[0, %34] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i64>
//       scf.if %arg0 {
//         llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], 512, [$2];", "l,l,r,l" %33, %28, %35 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
//       }
//     }
//     scf.yield %12, %22 : index, f32
//   }
//   return %10#1 : f32
// }

