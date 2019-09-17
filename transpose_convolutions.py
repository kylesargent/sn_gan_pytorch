import torch
import torch.nn.functional as F
import tqdm
import torch.nn as nn

def make_toep(input_size_h, input_size_w, kernel):
    input_size = input_size_h * input_size_w
    kernel_size_h, kernel_size_w = tuple(kernel.shape)
    
    output_size_h = (input_size_h - kernel_size_h + 1)
    output_size_w = (input_size_w - kernel_size_w + 1)
    output_size = output_size_h * output_size_w
    
    toep = torch.zeros(output_size, input_size)
    # for each row in the resized output
    for row in range(output_size_h * output_size_w):
        # get upper-left corner of convolutional window
        i = row // output_size_w
        j = row % output_size_w

        # get corresponding location in input vector
        offset = i * input_size_w + j

        # map kernel weights onto toeplitz w
        for ki in range(kernel_size_h):
            for kj in range(kernel_size_w):
                k = kernel[ki][kj]
                toep[row][offset + ki * input_size_w + kj] = k
    return toep

def make_block_toep(input_size_h, input_size_w, kernels):
    in_channels = kernels.shape[0]
    out_channels = kernels.shape[1]
    
    out_blocks = []
    for out_kernel in (range(out_channels)):
        in_blocks = []
        for in_kernel in range(in_channels):
            kernel = kernels[in_kernel][out_kernel]
            in_block = make_toep(input_size_h, input_size_w, kernel)
            in_blocks.append(in_block)
        out_block = torch.cat(in_blocks, dim=0)
        out_blocks.append(out_block)
    block_toep = torch.cat(out_blocks, dim=1)
    return block_toep

def make_jacobian(inputs, outputs):
    grads = []
    for i in range(outputs.shape[0]):
        grad = torch.autograd.grad(
            outputs=outputs[i],
            inputs=inputs,
            retain_graph=True, 
            create_graph=True
        )
        grads += [grad[0].unsqueeze(1)]
    return torch.cat(grads, dim=1).transpose(0,1)

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def get_transposed_convolution_func(weight):
    transpose_weight = flip(flip(weight.transpose(0,1), 2), 3)

    # we don't support asymmetric kernel sizes
    assert(transpose_weight.shape[2] == transpose_weight.shape[3])
    kernel_size = transpose_weight.shape[2]

    return lambda input: F.conv2d(input, transpose_weight, bias=None, stride=1, padding=kernel_size - 1)

def test(conv, n_tests=10):
    assert(torch.sum(conv.bias**2) == 0)

    # we don't support asymmetric padding
    assert(conv.padding[0] == conv.padding[1])
    padding = conv.padding[0]
    in_channels = conv.in_channels
    out_channels = conv.out_channels

    input_size_h = 32
    input_size_w = 32

    toep = make_block_toep(input_size_h + 2 * padding, input_size_w + 2 * padding, conv.weight)
    tconv = get_transposed_convolution_func(conv.weight)
    
    for test in range(n_tests):
        # check toeplitz matrix is faithful as matrix multiplication for original conv
        inputs = torch.randn(1 * in_channels * input_size_h * input_size_w)
        h1 = inputs.view(1, in_channels, input_size_h, input_size_w)
        h2 = conv(h1)
        outputs = h2.view(-1)

        inputs = F.pad(h1, (padding, padding, padding, padding)).view(-1, 1)

        print(torch.matmul(toep, inputs).view(-1) - outputs)

        err = torch.norm(torch.matmul(toep, inputs).view(-1) - outputs)
        print(err)

        # check transposition function is accurate as multiplication by transpose of toep
        f = lambda inp: tconv(inp).view(-1)
        g = lambda inp: torch.matmul(toep.transpose(0,1), inp.view(-1, 1)).view(-1)
        err = torch.norm(f(h2) - g(h2))
        print(err, '\n')


