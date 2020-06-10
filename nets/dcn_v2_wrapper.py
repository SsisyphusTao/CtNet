import torch
from torch.autograd import Function, gradcheck
from torch.nn import Module
import dcn_op_v2

class DeformableConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias, offset, mask, stride, pad, dilation, deformable_groups):
        assert type(input_tensor)==type(weight)==type(offset)==type(mask)==type(bias)==torch.Tensor
        assert type(stride) == type(pad) == type(dilation)==tuple
        assert len(stride) == len(pad) == len(dilation)==len(bias)==2
        
        ctx.stride_h = stride[0]
        ctx.stride_w = stride[1]
        ctx.pad_h = pad[0]
        ctx.pad_w = pad[1]
        ctx.dilation_h = dilation[0]
        ctx.dilation_w = dilation[1]
        ctx.deformable_groups = deformable_groups

        output = dcn_op_v2.forward(
            input_tensor,
            weight,
            bias,
            offset,
            mask,
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.deformable_groups
        )
        ctx.save_for_backward(input_tensor, weight, offset, mask, bias)
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        input_tensor, weight, offset, mask, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias, grad_offset, grad_mask = dcn_op_v2.backward(
            input_tensor,
            weight,
            bias,
            offset,
            mask,
            grad_outputs[0],
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w,
            ctx.deformable_groups
        )
        
        return grad_input, grad_weight, grad_bias, grad_offset, grad_mask, \
            None, None, None, None

if __name__ == "__main__":
    deformable_groups = 1
    N, inC, inH, inW = 2, 2, 4, 4
    outC = 2
    kH, kW = 3, 3
    def check_gradient_dconv():

        t = torch.randn(N, inC, inH, inW).cuda()
        t.requires_grad = True

        offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda()
        # offset.data.zero_()
        # offset.data -= 0.5
        offset.requires_grad = True

        mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
        # mask.data.zero_()
        mask.requires_grad = True
        mask = torch.sigmoid(mask)

        weight = torch.randn(outC, inC, kH, kW).cuda()
        weight.requires_grad = True
        bias = torch.rand(outC).cuda()
        bias.requires_grad = True

        # DeformableConv2DFunction.apply(t, weight, bias, offset, mask, (1,1), (1,1), (1,1), deformable_groups)

        func = DeformableConv2DFunction.apply
        gradcheck(func, (t, weight, bias, offset, mask, (1,1), (1,1), (1,1), deformable_groups), eps=1e-3, atol=1e-3, rtol=1e-2)
    check_gradient_dconv()