// RUN: emitters_opt %s -allow-unregistered-dialect -xla-flatten-tensors --mlir-print-ir-before-all --mlir-print-ir-after-all 

// | FileCheck %s

func.func @test_async_copy(%input: tensor<2x3xf32>, %buffer: tensor<3x2x3xf32>, %membars: tensor<3xi64>, %slot: index) -> (tensor<3x2x3xf32>, tensor<3xi64>) {
    %out_buffers, %out_membars = "xla_gpu.async_copy_start"(%input, %buffer, %membars, %slot) : (tensor<2x3xf32>, tensor<3x2x3xf32>, tensor<3xi64>, index) ->
       (tensor<3x2x3xf32>, tensor<3xi64>)
    return %out_buffers, %out_membars : tensor<3x2x3xf32>, tensor<3xi64>
}

func.func @test_async_wait(%buffer: tensor<2x3x2xf32>, %membars: tensor<2xi64>, %slot: index) -> (tensor<3x2xf32>, tensor<2xi64>) {
   %out_element, %out_membars = "xla_gpu.async_copy_wait"(%buffer, %membars, %slot) : (tensor<2x3x2xf32>, tensor<2xi64>, index) -> (tensor<3x2xf32>, tensor<2xi64>) 
   return %out_element, %out_membars : tensor<3x2xf32>, tensor<2xi64>
}


//func.func @test_async_copy2(%input: tensor<6xf32>, %buffer: tensor<18xf32>, %membars: tensor<3xi64>, %slot: index) -> (tensor<18xf32>, tensor<3xi64>) {
//    %out_buffers, %out_membars = "xla_gpu.async_copy_start"(%input, %buffer, %membars, %slot) : (tensor<6xf32>, tensor<18xf32>, tensor<3xi64>, index) ->
//       (tensor<18xf32>, tensor<3xi64>)
//       "dummy.use"(%out_buffers) : (tensor<18xf32>) -> ()
//    return %out_buffers, %out_membars : tensor<18xf32>, tensor<3xi64>
//}

