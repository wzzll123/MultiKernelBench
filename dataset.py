category2exampleop = {
    "matmul": "matmul_add",
    "activation": "leaky_relu",
    "loss": "mse_loss",
    "normalization": "layer_norm",
    'reduce': 'reduce_sum'
}
dataset = {
    "log_softmax": {
        "category": "activation"
    },
    "softsign": {
        "category": "activation"
    },
    "relu": {
        "category": "activation"
    },
    "elu": {
        "category": "activation"
    },
    "softplus": {
        "category": "activation"
    },
    "softmax": {
        "category": "activation"
    },
    "selu": {
        "category": "activation"
    },
    "min_gpt_new_gelu": {
        "category": "activation"
    },
    "gelu": {
        "category": "activation"
    },
    "tanh": {
        "category": "activation"
    },
    "sigmoid": {
        "category": "activation"
    },
    "hardsigmoid": {
        "category": "activation"
    },
    "swish": {
        "category": "activation"
    },
    "leaky_relu": {
        "category": "activation"
    },
    "hardtanh": {
        "category": "activation"
    },
    "where_broadcast": {
        "category": "broadcast"
    },
    "logic_and_broadcast": {
        "category": "broadcast"
    },
    "power_broadcast": {
        "category": "broadcast"
    },
    "max_broadcast": {
        "category": "broadcast"
    },
    "clamp_broadcast": {
        "category": "broadcast"
    },
    "add_bias_broadcast": {
        "category": "broadcast"
    },
    "add_bias_four_dim_broadcast": {
        "category": "broadcast"
    },
    "elmentwise_mul_broadcast": {
        "category": "broadcast"
    },
    "division_broadcast": {
        "category": "broadcast"
    },
    "subtract_with_bias_broadcast": {
        "category": "broadcast"
    },
    "conv_standard_3d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_depthwise_separable_2d": {
        "category": "convolution"
    },
    "conv_standard_2d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_1d_dilated": {
        "category": "convolution"
    },
    "conv_transposed_1d": {
        "category": "convolution"
    },
    "conv_pointwise_2d": {
        "category": "convolution"
    },
    "conv_standard_2d_square_input_asymmetric_kernel_dilated_padded": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided": {
        "category": "convolution"
    },
    "conv_transposed_2d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped": {
        "category": "convolution"
    },
    "conv_transposed_3d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_depthwise_2d_square_input_square_kernel": {
        "category": "convolution"
    },
    "conv_standard_3d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_standard_2d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_standard_2d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated": {
        "category": "convolution"
    },
    "conv_standard_3d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_standard_3d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_standard_1d_dilated_strided": {
        "category": "convolution"
    },
    "conv_depthwise_2d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_depthwise_2d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded": {
        "category": "convolution"
    },
    "conv_standard_1d": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped": {
        "category": "convolution"
    },
    "conv_depthwise_2d_square_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_asymmetric_input_square_kernel": {
        "category": "convolution"
    },
    "conv_transposed_3d_square_input_square_kernel_padded_dilated_strided": {
        "category": "convolution"
    },
    "conv_standard_2d_asymmetric_input_asymmetric_kernel": {
        "category": "convolution"
    },
    "lenet5": {
        "category": "arch"
    },
    "unet_softmax": {
        "category": "arch"
    },
    "vision_attention": {
        "category": "arch"
    },
    "densenet121_transition_layer": {
        "category": "arch"
    },
    "shallow_wide_mlp": {
        "category": "arch"
    },
    "mini_gpt_block": {
        "category": "arch"
    },
    "mamba_return_final_state": {
        "category": "arch"
    },
    "shufflenet_unit": {
        "category": "arch"
    },
    "swin_mlp": {
        "category": "arch"
    },
    "lstm_cn": {
        "category": "arch"
    },
    "lstm_hn": {
        "category": "arch"
    },
    "vision_transformer": {
        "category": "arch"
    },
    "net_vlad_with_ghost_clusters": {
        "category": "arch"
    },
    "net_vlad_no_ghost_clusters": {
        "category": "arch"
    },
    "relu_self_attention": {
        "category": "arch"
    },
    "mobilenet_v1": {
        "category": "arch"
    },
    "mamba_return_y": {
        "category": "arch"
    },
    "regnet": {
        "category": "arch"
    },
    "mlp": {
        "category": "arch"
    },
    "efficientnet_mb_conv": {
        "category": "arch"
    },
    "densenet121_dense_block": {
        "category": "arch"
    },
    "swintransformer_v2": {
        "category": "arch"
    },
    "densenet121": {
        "category": "arch"
    },
    "googlenet_inception_v1": {
        "category": "arch"
    },
    "squeeze_net_fire_module": {
        "category": "arch"
    },
    "resnet101": {
        "category": "arch"
    },
    "squeeze_net": {
        "category": "arch"
    },
    "vgg16": {
        "category": "arch"
    },
    "efficientnet_b0": {
        "category": "arch"
    },
    "lstm_bidirectional": {
        "category": "arch"
    },
    "deep_narrow_mlp": {
        "category": "arch"
    },
    "gru": {
        "category": "arch"
    },
    "resnet18": {
        "category": "arch"
    },
    "lstm": {
        "category": "arch"
    },
    "vanilla_rnn_hidden": {
        "category": "arch"
    },
    "googlenet_inception_module": {
        "category": "arch"
    },
    "efficientnet_b2": {
        "category": "arch"
    },
    "gru_bidirectional_hidden": {
        "category": "arch"
    },
    "shufflenet": {
        "category": "arch"
    },
    "gru_birectional": {
        "category": "arch"
    },
    "mobilenet_v2": {
        "category": "arch"
    },
    "vgg19": {
        "category": "arch"
    },
    "alexnet": {
        "category": "arch"
    },
    "gru_hidden": {
        "category": "arch"
    },
    "efficientnet_b1": {
        "category": "arch"
    },
    "min_gpt_causal_attention": {
        "category": "arch"
    },
    "vanilla_rnn": {
        "category": "arch"
    },
    "densenet201": {
        "category": "arch"
    },
    "resnet_basic_block": {
        "category": "arch"
    },
    "convolutional_vision_transformer": {
        "category": "arch"
    },
    "convtranspose3d_relu_groupnorm": {
        "category": "fuse"
    },
    "conv2d_subtract_hard_swish_max_pool_mish": {
        "category": "fuse"
    },
    "conv_transpose3d_batch_norm_avg_pool_avg_pool": {
        "category": "fuse"
    },
    "conv3d_divide_max_global_avg_pool_bias_add_sum": {
        "category": "fuse"
    },
    "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu": {
        "category": "fuse"
    },
    "conv3d_hardswish_relu_softmax_mean": {
        "category": "fuse"
    },
    "conv2d_min_add_multiply": {
        "category": "fuse"
    },
    "conv_transpose2d_gelu_group_norm": {
        "category": "fuse"
    },
    "conv_transpose2d_add_min_gelu_multiply": {
        "category": "fuse"
    },
    "matmul_divide_gelu": {
        "category": "fuse"
    },
    "conv2d_relu_hard_swish": {
        "category": "fuse"
    },
    "conv2d_tanh_scaling_bias_add_max": {
        "category": "fuse"
    },
    "conv_transpose3d_multiply_max_global_avg_pool_clamp": {
        "category": "fuse"
    },
    "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp": {
        "category": "fuse"
    },
    "gemm_divide_sum_scaling": {
        "category": "fuse"
    },
    "conv2d_batch_norm_scaling": {
        "category": "fuse"
    },
    "conv_transpose3d_avg_pool_clamp_softmax_multiply": {
        "category": "fuse"
    },
    "conv_transpose2d_bias_add_clamp_scaling_clamp_divide": {
        "category": "fuse"
    },
    "conv3d_multiply_instance_norm_clamp_multiply_max": {
        "category": "fuse"
    },
    "conv_transpose3d_layer_norm_gelu_scaling": {
        "category": "fuse"
    },
    "conv3d_group_norm_min_clamp_dropout": {
        "category": "fuse"
    },
    "gemm_group_norm_swish_multiply_swish": {
        "category": "fuse"
    },
    "conv2d_subtract_tanh_subtract_avg_pool": {
        "category": "fuse"
    },
    "matmul_swish_scaling": {
        "category": "fuse"
    },
    "conv2d_gelu_global_avg_pool": {
        "category": "fuse"
    },
    "matmul_min_subtract": {
        "category": "fuse"
    },
    "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean": {
        "category": "fuse"
    },
    "conv_transpose2d_max_pool_hardtanh_mean_tanh": {
        "category": "fuse"
    },
    "matmul_subtract_multiply_relu": {
        "category": "fuse"
    },
    "conv_transpose2d_subtract_tanh": {
        "category": "fuse"
    },
    "matmul_swish_sum_group_norm": {
        "category": "fuse"
    },
    "conv3d_max_log_sum_exp_relu": {
        "category": "fuse"
    },
    "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp": {
        "category": "fuse"
    },
    "conv_transpose3d_sum_layer_norm_avg_pool_gelu": {
        "category": "fuse"
    },
    "conv2d_avg_pool_sigmoid_sum": {
        "category": "fuse"
    },
    "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add": {
        "category": "fuse"
    },
    "gemm_group_norm_min_bias_add": {
        "category": "fuse"
    },
    "conv3d_softmax_max_pool_max_pool": {
        "category": "fuse"
    },
    "gemm_group_norm_hardtanh": {
        "category": "fuse"
    },
    "conv2d_group_norm_scale_max_pool_clamp": {
        "category": "fuse"
    },
    "gemm_swish_divide_clamp_tanh_clamp": {
        "category": "fuse"
    },
    "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply": {
        "category": "fuse"
    },
    "conv2d_divide_leaky_relu": {
        "category": "fuse"
    },
    "matmul_dropout_mean_softmax": {
        "category": "fuse"
    },
    "conv_transpose3d_swish_group_norm_hard_swish": {
        "category": "fuse"
    },
    "conv2d_instance_norm_divide": {
        "category": "fuse"
    },
    "conv2d_scaling_min": {
        "category": "fuse"
    },
    "conv_transpose3d_scaling_avg_pool_bias_add_scaling": {
        "category": "fuse"
    },
    "gemm_max_subtract_gelu": {
        "category": "fuse"
    },
    "gemm_batch_norm_scaling_softmax": {
        "category": "fuse"
    },
    "conv2d_multiply_leaky_relu_gelu": {
        "category": "fuse"
    },
    "conv_transpose3d_batch_norm_subtract": {
        "category": "fuse"
    },
    "convtranspose2d_batchnorm_tanh_maxpool_groupnorm": {
        "category": "fuse"
    },
    "conv2d_activation_batch_norm": {
        "category": "fuse"
    },
    "gemm_scale_batch_norm": {
        "category": "fuse"
    },
    "conv_transpose3d_sum_residual_add_multiply_residual_add": {
        "category": "fuse"
    },
    "conv_transpose3d_clamp_min_divide": {
        "category": "fuse"
    },
    "gemm_scaling_hard_tanh_gelu": {
        "category": "fuse"
    },
    "matmul_scale_residual_add_clamp_log_sum_exp_mish": {
        "category": "fuse"
    },
    "matmul_scaling_residual_add": {
        "category": "fuse"
    },
    "bmm_instance_norm_sum_residual_add_multiply": {
        "category": "fuse"
    },
    "conv_transpose3d_max_pool_softmax_subtract_swish_max": {
        "category": "fuse"
    },
    "conv2d_mish_mish": {
        "category": "fuse"
    },
    "gemm_add_relu": {
        "category": "fuse"
    },
    "gemm_relu_divide": {
        "category": "fuse"
    },
    "conv3d_leaky_relu_sum_clamp_gelu": {
        "category": "fuse"
    },
    "matmul_group_norm_leaky_relu_sum": {
        "category": "fuse"
    },
    "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max": {
        "category": "fuse"
    },
    "gemm_sigmoid_sum_log_sum_exp": {
        "category": "fuse"
    },
    "conv_transpose3d_add_hard_swish": {
        "category": "fuse"
    },
    "conv2d_min_tanh_tanh": {
        "category": "fuse"
    },
    "conv2d_relu_bias_add": {
        "category": "fuse"
    },
    "gemm_scale_batchnorm": {
        "category": "fuse"
    },
    "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add": {
        "category": "fuse"
    },
    "matmul_add_swish_tanh_gelu_hardtanh": {
        "category": "fuse"
    },
    "matmul_sigmoid_sum": {
        "category": "fuse"
    },
    "conv3d_mish_tanh": {
        "category": "fuse"
    },
    "matmul_batch_norm_bias_add_divide_swish": {
        "category": "fuse"
    },
    "conv2d_subtract_subtract_mish": {
        "category": "fuse"
    },
    "conv_transpose3d_scale_batch_norm_global_avg_pool": {
        "category": "fuse"
    },
    "gemm_bias_add_hardtanh_mish_group_norm": {
        "category": "fuse"
    },
    "conv3d_scaling_tanh_multiply_sigmoid": {
        "category": "fuse"
    },
    "conv_transpose3d_softmax_sigmoid": {
        "category": "fuse"
    },
    "matmul_gelu_softmax": {
        "category": "fuse"
    },
    "conv2d_add_scale_sigmoid_group_norm": {
        "category": "fuse"
    },
    "matmul_avg_pool_gelu_scale_max": {
        "category": "fuse"
    },
    "convtranspose2d_softmax_biasadd_scaling_sigmoid": {
        "category": "fuse"
    },
    "matmul_mish_mish": {
        "category": "fuse"
    },
    "matmul_max_pool_sum_scale": {
        "category": "fuse"
    },
    "conv_transpose3d_leaky_relu_multiply_leaky_relu_max": {
        "category": "fuse"
    },
    "gemm_multiply_leakyrelu": {
        "category": "fuse"
    },
    "gemm_sigmoid_scaling_residual_add": {
        "category": "fuse"
    },
    "conv_transpose2d_mish_add_hardtanh_scaling": {
        "category": "fuse"
    },
    "gemm_batch_norm_gelu_group_norm_mean_relu": {
        "category": "fuse"
    },
    "conv3d_group_norm_mean": {
        "category": "fuse"
    },
    "conv2d_hard_swish_relu": {
        "category": "fuse"
    },
    "convtranspose3d_mean_add_softmax_tanh_scaling": {
        "category": "fuse"
    },
    "conv_transpose3d_max_max_sum": {
        "category": "fuse"
    },
    "conv_transpose2d_min_sum_gelu_add": {
        "category": "fuse"
    },
    "conv3d_min_softmax": {
        "category": "fuse"
    },
    "triplet_margin_loss": {
        "category": "loss"
    },
    "kl_div_loss": {
        "category": "loss"
    },
    "cosine_similarity_loss": {
        "category": "loss"
    },
    "huber_loss": {
        "category": "loss"
    },
    "mse_loss": {
        "category": "loss"
    },
    "cross_entropy_loss": {
        "category": "loss"
    },
    "hinge_loss": {
        "category": "loss"
    },
    "cumsum_exclusive": {
        "category": "math"
    },
    "cumprod": {
        "category": "math"
    },
    "masked_cumsum": {
        "category": "math"
    },
    "matrix_scalar_multiplication": {
        "category": "math"
    },
    "cumsum": {
        "category": "math"
    },
    "cumsum_reverse": {
        "category": "math"
    },
    "matmul_with_diagonal_matrices": {
        "category": "matmul"
    },
    "matmul_with_transposed_a": {
        "category": "matmul"
    },
    "matmul_for_lower_triangular_matrices": {
        "category": "matmul"
    },
    "batched_matrix_multiplication": {
        "category": "matmul"
    },
    "square_matrix_multiplication": {
        "category": "matmul"
    },
    "matmul_with_irregular_shapes": {
        "category": "matmul"
    },
    "four_dim_tensor_matrix_multiplication": {
        "category": "matmul"
    },
    "tall_skinny_matrix_multiplication": {
        "category": "matmul"
    },
    "three_dim_tensor_matrix_multiplication": {
        "category": "matmul"
    },
    "matmul_with_large_k_dimension": {
        "category": "matmul"
    },
    "matmul_with_small_k_dimension": {
        "category": "matmul"
    },
    "standard_matrix_multiplication": {
        "category": "matmul"
    },
    "matmul_with_transposed_b": {
        "category": "matmul"
    },
    "matrix_vector_multiplication": {
        "category": "matmul"
    },
    "matmul_with_transposed_both": {
        "category": "matmul"
    },
    "matmul_for_symmetric_matrices": {
        "category": "matmul"
    },
    "matmul_for_upper_triangular_matrices": {
        "category": "matmul"
    },
    "rms_norm": {
        "category": "normalization"
    },
    "l2_norm": {
        "category": "normalization"
    },
    "l1_norm": {
        "category": "normalization"
    },
    "frobenius_norm": {
        "category": "normalization"
    },
    "group_norm": {
        "category": "normalization"
    },
    "batch_norm": {
        "category": "normalization"
    },
    "instance_norm": {
        "category": "normalization"
    },
    "layer_norm": {
        "category": "normalization"
    },
    "adam": {
        "category": "optimizer"
    },
    "adagrad": {
        "category": "optimizer"
    },
    "lamb": {
        "category": "optimizer"
    },
    "rmsprop": {
        "category": "optimizer"
    },
    "sgd": {
        "category": "optimizer"
    },
    "average_pooling_3d": {
        "category": "pooling"
    },
    "max_pooling_1d": {
        "category": "pooling"
    },
    "max_pooling_3d": {
        "category": "pooling"
    },
    "average_pooling_2d": {
        "category": "pooling"
    },
    "average_pooling_1d": {
        "category": "pooling"
    },
    "max_pooling_2d": {
        "category": "pooling"
    },
    "index_select": {
        "category": "index"
    },
    "inplace_update": {
        "category": "index"
    },
    "argmax_over_a_dimension": {
        "category": "index"
    },
    "gather": {
        "category": "index"
    },
    "scatter": {
        "category": "index"
    },
    "index_copy": {
        "category": "index"
    },
    "masked_fill": {
        "category": "index"
    },
    "index_add": {
        "category": "index"
    },
    "embedding": {
        "category": "index"
    },
    "scatter_add": {
        "category": "index"
    },
    "take_along_dim": {
        "category": "index"
    },
    "argmin_over_a_dimension": {
        "category": "index"
    },
    "bicubic_upsample": {
        "category": "resize"
    },
    "nearest_neighbor_upsample": {
        "category": "resize"
    },
    "downsample_bilinear": {
        "category": "resize"
    },
    "resize_with_antialias": {
        "category": "resize"
    },
    "interpolate_dynamic": {
        "category": "resize"
    },
    "upsample_grid_sample": {
        "category": "resize"
    },
    "bilinear_upsample": {
        "category": "resize"
    },
    "grid_sample_random_warp": {
        "category": "resize"
    },
    "grid_sample_affine": {
        "category": "resize"
    },
    "trilinear_upsample": {
        "category": "resize"
    },
    "sum_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "product_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "mean_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "max_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "min_reduction_over_a_dimension": {
        "category": "reduce"
    },
    "causal_attention": {
        "category": "attention"
    },
    "kv_cached_attention_inference": {
        "category": "attention"
    },
    "multi_head_attention": {
        "category": "attention"
    },
    "scaled_dot_product_attention_long_context": {
        "category": "attention"
    },
    "cross_attention": {
        "category": "attention"
    },
    "kv_cached_chat_batch_attention": {
        "category": "attention"
    },
    "multi_query_attention": {
        "category": "attention"
    },
    "sparse_attention": {
        "category": "attention"
    },
    "cross_modal_attention": {
        "category": "attention"
    },
    "kv_cached_speculative_attention": {
        "category": "attention"
    },
    "scaled_dot_product_attention": {
        "category": "attention"
    },
    "windowed_causal_attention": {
        "category": "attention"
    },
    "group_query_attention": {
        "category": "attention"
    },
    "linear_attention": {
        "category": "attention"
    },
    "scaled_dot_product_attention_inference": {
        "category": "attention"
    }
}
