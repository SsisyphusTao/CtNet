#ifndef DCN_V2_CUDA
#define DCN_V2_CUDA

at::Tensor          dcn_v2_cuda_forward(torch::Tensor& input, torch::Tensor& weight,
                                        torch::Tensor& bias,
                                        torch::Tensor& offset, torch::Tensor& mask,
                                        //  int kernel_h, int kernel_w,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w,
                                        const int dilation_h, const int dilation_w,
                                        const int deformable_group);
vector<at::Tensor>  dcn_v2_cuda_backward(torch::Tensor& input, torch::Tensor& weight,
                                         torch::Tensor& bias,
                                         torch::Tensor& offset, torch::Tensor& mask,                           
                                        //  torch::Tensor& grad_input, torch::Tensor& grad_weight,
                                        //  torch::Tensor& grad_bias, torch::Tensor& grad_offset,
                                        //  torch::Tensor& grad_mask, 
                                         torch::Tensor& grad_output,
                                        //  int kernel_h, int kernel_w,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w,
                                         int dilation_h, int dilation_w,
                                         int deformable_group);

#endif