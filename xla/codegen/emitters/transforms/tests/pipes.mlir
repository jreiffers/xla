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
//   %c512_i32 = arith.constant 512 : i32
//   %c0_i64 = arith.constant 0 : i64
//   %false = arith.constant false
//   %c1_i32 = arith.constant 1 : i32
//   %true = arith.constant true
//   %2 = llvm.addrspacecast %0 : !llvm.ptr<3> to !llvm.ptr
//   %3 = llvm.addrspacecast %1 : !llvm.ptr<3> to !llvm.ptr
//   scf.if %arg0 {                                                                                                                                                                   [116/1844]
//     %20 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %20, %c128_i32 : !llvm.ptr, i32
//     %21 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %21, %c128_i32 : !llvm.ptr, i32
//     %22 = llvm.getelementptr inbounds %3[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %22, %c128_i32 : !llvm.ptr, i32
//     %23 = llvm.getelementptr inbounds %3[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//     nvvm.mbarrier.init %23, %c128_i32 : !llvm.ptr, i32
//   }
//   gpu.barrier
//   %4 = llvm.getelementptr inbounds %arg2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x f32>
//   %5 = llvm.getelementptr inbounds %arg2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x f32>
//   %6 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %7 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   scf.if %arg0 {
//     %20 = llvm.addrspacecast %7 : !llvm.ptr to !llvm.ptr<3>
//     %21 = llvm.addrspacecast %6 : !llvm.ptr to !llvm.ptr<3>
//     func.call @llvm.nvvm.cp.async.bulk.global.to.shared.cta(%21, %20, %4, %c512_i32, %c0_i64, %false) : (!llvm.ptr<3>, !llvm.ptr<3>, !llvm.ptr, i32, i64, i1) -> ()
//   }
//   %8 = llvm.getelementptr inbounds %2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %9 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   scf.if %arg0 {
//     %20 = llvm.addrspacecast %9 : !llvm.ptr to !llvm.ptr<3>
//     %21 = llvm.addrspacecast %8 : !llvm.ptr to !llvm.ptr<3>
//     func.call @llvm.nvvm.cp.async.bulk.global.to.shared.cta(%21, %20, %5, %c512_i32, %c0_i64, %false) : (!llvm.ptr<3>, !llvm.ptr<3>, !llvm.ptr, i32, i64, i1) -> ()
//   }
//   %10 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %11 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   %12 = llvm.addrspacecast %11 : !llvm.ptr to !llvm.ptr<3>
//   %13 = scf.if %arg0 -> (i64) {
//     %20 = func.call @llvm.nvvm.mbarrier.arrive.expect.tx.relaxed.scope.cta.space.cta(%12, %c512_i32) : (!llvm.ptr<3>, i32) -> i64
//     scf.yield %20 : i64
//   } else {
//     %20 = func.call @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(%12, %c1_i32) : (!llvm.ptr<3>, i32) -> i64
//     scf.yield %20 : i64
//   }
//   %14 = scf.while (%arg3 = %true) : (i1) -> i1 {
//     %20 = func.call @llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta(%12, %13) : (!llvm.ptr<3>, i64) -> i1
//     scf.condition(%20) %20 : i1
//   } do {
//   ^bb0(%arg3: i1):
//     scf.yield %arg3 : i1
//   }
//   %15 = llvm.getelementptr inbounds %2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<512 x f32>
//   %16 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i64>
//   %17 = llvm.addrspacecast %16 : !llvm.ptr to !llvm.ptr<3>
//   %18 = scf.if %arg0 -> (i64) {
//     %20 = func.call @llvm.nvvm.mbarrier.arrive.expect.tx.relaxed.scope.cta.space.cta(%17, %c512_i32) : (!llvm.ptr<3>, i32) -> i64
//     scf.yield %20 : i64
//   } else {
//     %20 = func.call @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(%17, %c1_i32) : (!llvm.ptr<3>, i32) -> i64
//     scf.yield %20 : i64
//   }
//   %19 = scf.while (%arg3 = %true) : (i1) -> i1 {
//     %20 = func.call @llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta(%17, %18) : (!llvm.ptr<3>, i64) -> i1
//     scf.condition(%20) %20 : i1
//   } do {
//   ^bb0(%arg3: i1):
//     scf.yield %arg3 : i1
//   }
//   "dummy.consume"(%10, %15) : (!llvm.ptr, !llvm.ptr) -> ()
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
//   %c512_i32 = arith.constant 512 : i32
//   %c0_i64 = arith.constant 0 : i64
//   %false = arith.constant false
//   %c1_i32 = arith.constant 1 : i32
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
//   %4 = llvm.getelementptr inbounds %arg2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1280 x f32>                                                                                  [27/1844]
//   %5 = llvm.getelementptr inbounds %arg2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1280 x f32>
//   %6 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x f32>
//   %7 = llvm.getelementptr inbounds %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
//   scf.if %arg0 {
//     %11 = llvm.addrspacecast %7 : !llvm.ptr to !llvm.ptr<3>
//     %12 = llvm.addrspacecast %6 : !llvm.ptr to !llvm.ptr<3>
//     func.call @llvm.nvvm.cp.async.bulk.global.to.shared.cta(%12, %11, %4, %c512_i32, %c0_i64, %false) : (!llvm.ptr<3>, !llvm.ptr<3>, !llvm.ptr, i32, i64, i1) -> ()
//   }
//   %8 = llvm.getelementptr inbounds %2[0, 128] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x f32>
//   %9 = llvm.getelementptr inbounds %3[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i64>
//   scf.if %arg0 {
//     %11 = llvm.addrspacecast %9 : !llvm.ptr to !llvm.ptr<3>
//     %12 = llvm.addrspacecast %8 : !llvm.ptr to !llvm.ptr<3>
//     func.call @llvm.nvvm.cp.async.bulk.global.to.shared.cta(%12, %11, %5, %c512_i32, %c0_i64, %false) : (!llvm.ptr<3>, !llvm.ptr<3>, !llvm.ptr, i32, i64, i1) -> ()
//   }
//   %10:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %c0, %arg5 = %cst) -> (index, f32) {
//     %11 = arith.addi %arg4, %c1 : index
//     %12 = arith.remui %11, %c2 : index
//     %13 = arith.muli %arg4, %c128 : index
//     %14 = arith.index_castui %13 : index to i64
//     %15 = llvm.getelementptr inbounds %2[0, %14] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x f32>
//     %16 = arith.index_castui %arg4 : index to i64
//     %17 = llvm.getelementptr inbounds %3[0, %16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i64>
//     %18 = llvm.addrspacecast %17 : !llvm.ptr to !llvm.ptr<3>
//     %19 = scf.if %arg0 -> (i64) {
//       %27 = func.call @llvm.nvvm.mbarrier.arrive.expect.tx.relaxed.scope.cta.space.cta(%18, %c512_i32) : (!llvm.ptr<3>, i32) -> i64
//       scf.yield %27 : i64
//     } else {
//       %27 = func.call @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(%18, %c1_i32) : (!llvm.ptr<3>, i32) -> i64
//       scf.yield %27 : i64
//     }
//     %20 = scf.while (%arg6 = %true) : (i1) -> i1 {
//       %27 = func.call @llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta(%18, %19) : (!llvm.ptr<3>, i64) -> i1
//       scf.condition(%27) %27 : i1
//     } do {
//     ^bb0(%arg6: i1):
//       scf.yield %arg6 : i1
//     }
//     %21 = llvm.getelementptr inbounds %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x f32>
//     %22 = llvm.load %21 : !llvm.ptr -> f32
//     %23 = arith.addf %arg5, %22 : f32
//     %24 = arith.addi %arg3, %c2 : index
//     %25 = arith.muli %24, %c128 : index
//     %26 = arith.cmpi sge, %arg3, %c8 : index
//     scf.if %26 {
//     } else {
//       %27 = xla.apply_indexing #indexing_map(%25)
//       %28 = arith.index_castui %27 : index to i64
//       %29 = llvm.getelementptr inbounds %arg2[0, %28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1280 x f32>
//       %30 = arith.addi %12, %c1 : index
//       %31 = arith.remui %30, %c2 : index
//       %32 = arith.muli %31, %c128 : index
//       %33 = arith.index_castui %32 : index to i64
//       %34 = llvm.getelementptr inbounds %2[0, %33] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x f32>
//       %35 = arith.index_castui %31 : index to i64
//       %36 = llvm.getelementptr inbounds %3[0, %35] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i64>
//       scf.if %arg0 {
//         %37 = llvm.addrspacecast %36 : !llvm.ptr to !llvm.ptr<3>
//         %38 = llvm.addrspacecast %34 : !llvm.ptr to !llvm.ptr<3>
//         func.call @llvm.nvvm.cp.async.bulk.global.to.shared.cta(%38, %37, %29, %c512_i32, %c0_i64, %false) : (!llvm.ptr<3>, !llvm.ptr<3>, !llvm.ptr, i32, i64, i1) -> ()
//       }
//     }
//     scf.yield %12, %23 : index, f32
//   }
//   return %10#1 : f32
// }
