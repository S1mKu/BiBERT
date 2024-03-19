import math
import pdb

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, "already initialized."
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method="default"):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == "default":
            init_val = (
                2 * tensor.abs().mean() / math.sqrt(Qp)
                if symmetric
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
            )
        elif init_method == "uniform":
            init_val = 1.0 / (2 * Qp + 1) if symmetric else 1.0 / Qp

        self._initialize(init_val)


class ElasticQuantBinarizerUnsignedSoftmaxUnclipped(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(
                input, num_bits, symmetric=False, init_method="default"
            )
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, "alpha = {:.6f} becomes non-positive".format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other

        # norm to [0, 1 / alpha] since we try to minimize ||W_real - alpha * W_bin||
        q_w = input_ / alpha

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = (
            (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w + q_w.round())
                )
                * grad_output
                * grad_scale
            )
            .sum()
            .unsqueeze(dim=0)
        )
        # grad_input = indicate_middle * grad_output
        grad_input = grad_output.clone()
        return grad_input, grad_alpha, None, None


class AttnProbBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, mask):
        ctx.other = mask, bias
        bias = bias.expand_as(input) * mask
        output = input + bias
        offset = torch.tensor(0.000_1).float().to(input.device)
        # offset = torch.tensor(0.001).float().to(input.device)
        output = torch.where(output < offset, offset, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        mask, bias = ctx.other
        grad_bias = torch.sum(grad_output * mask).unsqueeze(0)
        # grad_bias = torch.sum(grad_output).unsqueeze(0)

        if bias < 0 and grad_bias > 0:
            grad_bias = torch.tensor([0.0]).to(grad_output.device)

        return grad_input, grad_bias, None
    

class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out == -1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (
                    (
                        input.max(dim=-1, keepdim=True)[0]
                        - input.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2 ** num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(self, *kargs, bias=True, config=None, type=None):
        super(QuantizeLinear, self).__init__(*kargs, bias=True)
        self.quantize_act = config.quantize_act
        self.weight_bits = config.weight_bits
        self.quantize_act = config.quantize_act
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.register_buffer(
            "weight_clip_val", torch.tensor([-config.clip_val, config.clip_val])
        )
        self.init = True

        if self.quantize_act:
            self.input_bits = config.input_bits
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            elif self.input_bits == 2:
                self.act_quantizer = TwnQuantizer
            else:
                self.act_quantizer = SymQuantizer
            self.register_buffer(
                "act_clip_val", torch.tensor([-config.clip_val, config.clip_val])
            )
        self.register_parameter("scale", Parameter(torch.Tensor([0.0]).squeeze()))

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        self.scale = Parameter(
            (ba.norm() / torch.sign(ba).norm()).float().to(ba.device)
        )

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = (
                binary_weights_no_grad.detach()
                - cliped_weights.detach()
                + cliped_weights
            )
        else:
            weight = self.weight_quantizer.apply(
                self.weight, self.weight_clip_val, self.weight_bits, True
            )

        if self.input_bits == 1:
            binary_input_no_grad = torch.sign(input)
            cliped_input = torch.clamp(input, -1.0, 1.0)
            ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(
                input, self.act_clip_val, self.input_bits, True
            )

        out = nn.functional.linear(ba, weight)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self, *kargs, padding_idx=None, config=None, type=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx=padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer(
            "weight_clip_val", torch.tensor([-config.clip_val, config.clip_val])
        )

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = (
                binary_weights_no_grad.detach()
                - cliped_weights.detach()
                + cliped_weights
            )
        else:
            weight = self.weight_quantizer.apply(
                self.weight, self.weight_clip_val, self.weight_bits, self.layerwise
            )
        out = nn.functional.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out
