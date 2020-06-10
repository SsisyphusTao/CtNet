#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "dcn_v2_im2col_cuda.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace torch;
using namespace c10::cuda;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

Tensor dcn_v2_cuda_forward(Tensor& input, Tensor& weight,
                         Tensor& bias,
                         Tensor& offset, Tensor& mask,
                        //  int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group)
{
    // THCAssertSameGPU(Tensor_checkGPU(state, 8, input, weight, bias, ones, offset, mask, output, columns));
    // THArgCheck(Tensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    // THArgCheck(Tensor_isContiguous(state, weight), 2, "weight tensor has to be contiguous");
    
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    // const int channels_out = weight.size(0);
    // const int channels_kernel = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    // if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    //     c10::Error("Input shape and kernel shape wont match: (%d x %d vs %d x %d).", 
    //     kernel_h_, kernel_w, kernel_h_, kernel_w_);
    // if (channels != channels_kernel)
    //     c10::Error("Input shape and kernel channels wont match: (%d vs %d).", 
    //     channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // if (Tensor_nDimension(state, ones) != 2 ||
    //     Tensor_size(state, ones, 0) * Tensor_size(state, ones, 1) < height_out * width_out)
    // {
    //     // Resize plane and fill with ones...
    //     Tensor_resize2d(state, ones, height_out, width_out);
    //     Tensor_fill(state, ones, 1);
    // }

    // Tensor ones = input.new_full({height_out * width_out, 1}, 1);
    Tensor columns = input.new_empty({channels * kernel_h * kernel_w, 1 * height_out * width_out});
    // ones.to(input.device());
    columns.to(input.device());
    // bias.unsqueeze_(0);

    // resize output
    // Tensor_resize4d(state, output, batch, channels_out, height_out, width_out);
    // resize temporary columns
    // Tensor_resize2d(state, columns, channels * kernel_h * kernel_w, 1 * height_out * width_out);
    
    // Tensor *input_n = Tensor_new(state);
    // Tensor *offset_n = Tensor_new(state);
    // Tensor *mask_n = Tensor_new(state);
    // Tensor *output_n = Tensor_new(state);
    vector<Tensor> output_slice;

    for (int b = 0; b < batch; b++)
    {
        // Tensor_select(state, input_n, input, 0, b);
        // Tensor_select(state, offset_n, offset, 0, b);
        // Tensor_select(state, mask_n, mask, 0, b);
        // Tensor_select(state, output_n, output, 0, b);

        Tensor input_n = input.slice(0, b, b+1).squeeze_();
        Tensor offset_n = offset.slice(0, b, b+1).squeeze_();
        Tensor mask_n = mask.slice(0, b, b+1).squeeze_();

        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        // long m_ = channels_out;
        // long n_ = height_out * width_out;
        // long k_ = 1;
        // THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //                  Tensor_data(state, ones), k_,
        //                  Tensor_data(state, bias), k_, 0.0f,
        //                  Tensor_data(state, output_n), n_);

        modulated_deformable_im2col_cuda(getCurrentCUDAStream(),
                                         input_n.data_ptr<float>(), offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, columns.data_ptr<float>());

        //(k * m)  x  (m * n)
        // Y = WC
        // long m = channels_out;
        // long n = height_out * width_out;
        // long k = channels * kernel_h * kernel_w;
        // THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
        //                  Tensor_data(state, columns), n,
        //                  Tensor_data(state, weight), k, 1.0f,
        //                  Tensor_data(state, output_n), n);
        output_slice.push_back(at::mm(weight.flatten(1),columns).resize_({weight.size(0), height_out, width_out}));
    }
    // Tensor_free(state, input_n);
    // Tensor_free(state, offset_n);
    // Tensor_free(state, mask_n);
    // Tensor_free(state, output_n);
    at::TensorList output = at::TensorList(output_slice);
    return at::stack(output);
}

vector<at::Tensor> dcn_v2_cuda_backward(Tensor& input, Tensor& weight,
                          Tensor& bias, //Tensor& ones,
                          Tensor& offset, Tensor& mask,
                          //Tensor& columns,
                        //   Tensor& grad_input, Tensor& grad_weight,
                        //   Tensor& grad_bias, Tensor& grad_offset,
                        //   Tensor& grad_mask, 
                          Tensor& grad_output,
                        //   int kernel_h, int kernel_w,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group)
{
    // THCAssertSameGPU(Tensor_checkGPU(state, 13, input, weight, bias, ones, offset, mask, columns,
    //                                        grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output));
    // THArgCheck(Tensor_isContiguous(state, input), 1, "input tensor has to be contiguous");
    // THArgCheck(Tensor_isContiguous(state, weight), 2, "weight tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    // const int channels_out = weight.size(0);
    // const int channels_kernel = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    // if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    //     c10::Error("Input shape and kernel shape wont match: (%d x %d vs %d x %d).", 
    //     kernel_h_, kernel_w, kernel_h_, kernel_w_);
    // if (channels != channels_kernel)
    //     c10::Error("Input shape and kernel channels wont match: (%d vs %d).", 
    //     channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // if (Tensor_nDimension(state, ones) != 2 ||
    //     Tensor_size(state, ones, 0) * Tensor_size(state, ones, 1) < height_out * width_out)
    // {
    //     // Resize plane and fill with ones...
    //     Tensor_resize2d(state, ones, height_out, width_out);
    //     Tensor_fill(state, ones, 1.0f);
    // }

    // Tensor_resize4d(state, grad_input, batch, channels, height, width);
    // Tensor_resize2d(state, columns, channels * kernel_h * kernel_w, height_out * width_out);

    // Tensor *input_n = Tensor_new(state);
    // Tensor *offset_n = Tensor_new(state);
    // Tensor *mask_n = Tensor_new(state);

    // Tensor *grad_output_n = Tensor_new(state);
    // Tensor *grad_input_n = Tensor_new(state);
    // Tensor *grad_offset_n = Tensor_new(state);
    // Tensor *grad_mask_n = Tensor_new(state);
    vector<Tensor> grad_input_list;
    Tensor grad_weight = weight.new_zeros(weight.sizes()).flatten(1);
    Tensor grad_bias = bias.new_zeros(bias.sizes());
    vector<Tensor> grad_offset_list;
    vector<Tensor> grad_mask_list;

    for (int b = 0; b < batch; b++)
    {
        // Tensor_select(state, input_n, input, 0, b);
        // Tensor_select(state, offset_n, offset, 0, b);
        // Tensor_select(state, mask_n, mask, 0, b);
        // Tensor_select(state, grad_output_n, grad_output, 0, b);
        // Tensor_select(state, grad_input_n, grad_input, 0, b);
        // Tensor_select(state, grad_offset_n, grad_offset, 0, b);
        // Tensor_select(state, grad_mask_n, grad_mask, 0, b);

        Tensor input_n = input.slice(0, b, b+1).squeeze_();
        Tensor offset_n = offset.slice(0, b, b+1).squeeze_();
        Tensor mask_n = mask.slice(0, b, b+1).squeeze_();
        Tensor grad_output_n = grad_output.slice(0, b, b+1).squeeze_();
        Tensor grad_input_n = input_n.new_zeros(input_n.sizes());
        Tensor grad_offset_n = offset_n.new_zeros(offset_n.sizes());
        Tensor grad_mask_n = mask_n.new_zeros(mask_n.sizes());

        // long m = channels * kernel_h * kernel_w;
        // long n = height_out * width_out;
        // long k = channels_out;

        // THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
        //                  Tensor_data(state, grad_output_n), n,
        //                  Tensor_data(state, weight), m, 0.0f,
        //                  Tensor_data(state, columns), n);
        Tensor columns = at::mm(weight.flatten(1).t_(), grad_output_n.flatten(1));

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(getCurrentCUDAStream(),
                                               columns.data_ptr<float>(),
                                               input_n.data_ptr<float>(),
                                               offset_n.data_ptr<float>(),
                                               mask_n.data_ptr<float>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<float>(),
                                               grad_mask_n.data_ptr<float>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(getCurrentCUDAStream(),
                                         columns.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<float>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(getCurrentCUDAStream(),
                                         input_n.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data_ptr<float>());
        // long m_ = channels_out;
        // long n_ = channels * kernel_h * kernel_w;
        // long k_ = height_out * width_out;

        // THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //                  Tensor_data(state, columns), k_,
        //                  Tensor_data(state, grad_output_n), k_, 1.0f,
        //                  Tensor_data(state, grad_weight), n_);

        grad_weight.addmm_(grad_output_n.flatten(1), columns.flatten(1).t_());

        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
    //     THCudaBlas_Sgemv(state,
    //                      't',
    //                      k_, m_, 1.0f,
    //                      Tensor_data(state, grad_output_n), k_,
    //                      Tensor_data(state, ones), 1, 1.0f,
    //                      Tensor_data(state, grad_bias), 1);

        grad_input_list.push_back(grad_input_n);
        grad_offset_list.push_back(grad_offset_n);
        grad_mask_list.push_back(grad_mask_n);
    }

    // Tensor_free(state, input_n);
    // Tensor_free(state, offset_n);
    // Tensor_free(state, mask_n);

    // Tensor_free(state, grad_output_n);
    // Tensor_free(state, grad_input_n);
    // Tensor_free(state, grad_offset_n);
    // Tensor_free(state, grad_mask_n);
    vector<Tensor> output;
    output.push_back(at::stack(at::TensorList(grad_input_list)).resize_(input.sizes()));
    output.push_back(grad_weight.resize_(weight.sizes()));
    output.push_back(grad_bias);
    output.push_back(at::stack(at::TensorList(grad_offset_list)).resize_(offset.sizes()));
    output.push_back(at::stack(at::TensorList(grad_mask_list)).resize_(mask.sizes()));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcn_v2_cuda_forward, "DCN operator forward (CUDA)");
  m.def("backward", &dcn_v2_cuda_backward, "DCN operator backward (CUDA)");
}