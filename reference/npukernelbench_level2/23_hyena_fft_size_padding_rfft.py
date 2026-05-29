import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused FFT size padding and real FFT computation for Hyena convolution.
    Pads input to 2*seqlen for circular convolution, computes real FFT (rfft),
    and normalizes by fft_size.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Applies FFT size padding and real FFT computation for Hyena convolution.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, seqlen].
                              Supports float32.

        Returns:
            tuple: (x_freq_real, x_freq_imag)
                - x_freq_real (torch.Tensor): Real part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
                - x_freq_imag (torch.Tensor): Imaginary part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
        """
        batch, channels, seqlen = x.shape
        fft_size = 2 * seqlen

        x_f32 = x.to(torch.float32)

        x_freq = torch.fft.rfft(x_f32, n=fft_size)

        x_freq = x_freq / fft_size

        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        return x_freq_real, x_freq_imag


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
