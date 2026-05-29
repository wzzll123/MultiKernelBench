import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs LSTM computation using NPU accelerated npu_lstm.
    Computes DynamicRNN with LSTM cells.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                seqMask: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
                has_biases: bool, num_layers: int, dropout: float,
                train: bool, bidirectional: bool, batch_first: bool,
                flag_seq: bool, direction: bool):
        """
        Performs LSTM computation on NPU.

        Args:
            x (Tensor): 4D input tensor [FRACTAL_NZ format], dtype float16/float32.
            weight (Tensor): 4D weight tensor [FRACTAL_NZ_LSTM format].
            bias (Tensor): 1D bias tensor [ND format].
            seqMask (Tensor): Sequence mask tensor.
            h (Tensor): 4D initial hidden state tensor.
            c (Tensor): 4D initial cell state tensor.
            has_biases (bool): Whether bias exists.
            num_layers (int): Number of recurrent layers (currently only supports 1).
            dropout (float): Dropout probability (currently not supported).
            train (bool): Whether in training mode.
            bidirectional (bool): Whether LSTM is bidirectional (currently not supported).
            batch_first (bool): Whether input is (batch, seq, feature) (currently not supported).
            flag_seq (bool): Whether input is PackedSequence (currently not supported).
            direction (bool): True for REDIRECTIONAL, False for UNIDIRECTIONAL.

        Returns:
            tuple: (y, output_h, output_c, i, j, f, o, tanhct)
        """
        import torch_npu
        return torch_npu.npu_lstm(x, weight, bias, seqMask, h, c,
                                   has_biases, num_layers, dropout, train,
                                   bidirectional, batch_first, flag_seq, direction)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_lstm.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        x_info = inputs[0]
        weight_info = inputs[1]
        bias_info = inputs[2]
        seqMask_info = inputs[3]
        h_info = inputs[4]
        c_info = inputs[5]
        has_biases_info = inputs[6]
        num_layers_info = inputs[7]
        dropout_info = inputs[8]
        train_info = inputs[9]
        bidirectional_info = inputs[10]
        batch_first_info = inputs[11]
        flag_seq_info = inputs[12]
        direction_info = inputs[13]

        dtype = dtype_map[x_info["dtype"]]

        x = torch.randn(x_info["shape"], dtype=dtype)
        weight = torch.randn(weight_info["shape"], dtype=dtype)
        bias = torch.randn(bias_info["shape"], dtype=dtype)

        seqMask_dtype_map = {
            "float16": torch.float16,
            "int32": torch.int32,
        }
        seqMask_dtype = seqMask_dtype_map.get(seqMask_info["dtype"], torch.float16)
        seqMask = torch.randn(seqMask_info["shape"], dtype=seqMask_dtype)

        h = torch.randn(h_info["shape"], dtype=dtype)
        c = torch.randn(c_info["shape"], dtype=dtype)

        has_biases = has_biases_info["value"]
        num_layers = num_layers_info["value"]
        dropout = dropout_info["value"]
        train = train_info["value"]
        bidirectional = bidirectional_info["value"]
        batch_first = batch_first_info["value"]
        flag_seq = flag_seq_info["value"]
        direction = direction_info["value"]

        input_groups.append([x, weight, bias, seqMask, h, c,
                             has_biases, num_layers, dropout, train,
                             bidirectional, batch_first, flag_seq, direction])
    return input_groups


def get_init_inputs():
    return []
