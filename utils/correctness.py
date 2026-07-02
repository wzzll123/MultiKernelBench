import torch

from config import num_correct_trials, project_root_path, seed_num

VALUE_PROFILES = (
    "reference",
    "normal",
    "uniform_signed",
    "small_magnitude",
    "large_magnitude",
    "near_one",
    "sparse_first",
)

def set_seed(seed: int):
    torch.manual_seed(seed)
    # Backend packages such as torch_musa register their device API on torch.
    for backend_name in ("cuda", "musa", "xpu", "npu"):
        backend = getattr(torch, backend_name, None)
        if backend is not None and hasattr(backend, "manual_seed"):
            backend.manual_seed(seed)


def _to_device(values, device):
    return [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in values
    ]


def _is_float_tensor(value):
    return isinstance(value, torch.Tensor) and (
        torch.is_floating_point(value) or torch.is_complex(value)
    )


def _rand_like_real(value):
    if torch.is_complex(value):
        real = torch.rand(value.shape, dtype=value.real.dtype, device=value.device)
        imag = torch.rand(value.shape, dtype=value.real.dtype, device=value.device)
        return torch.complex(real, imag).to(value.dtype)
    return torch.rand_like(value)


def _randn_like_real(value):
    if torch.is_complex(value):
        real = torch.randn(value.shape, dtype=value.real.dtype, device=value.device)
        imag = torch.randn(value.shape, dtype=value.real.dtype, device=value.device)
        return torch.complex(real, imag).to(value.dtype)
    return torch.randn_like(value)


def _with_requires_grad(value, requires_grad):
    if isinstance(value, torch.Tensor) and (
        value.is_floating_point() or value.is_complex()
    ):
        value.requires_grad_(requires_grad)
    return value


def _profile_tensor(value, profile, tensor_index):
    if not _is_float_tensor(value):
        return value

    requires_grad = value.requires_grad
    if profile == "reference":
        return value
    if profile == "normal":
        return _with_requires_grad(_randn_like_real(value), requires_grad)
    if profile == "uniform_signed":
        profiled = _rand_like_real(value) * 2 - 1
        return _with_requires_grad(profiled.to(value.dtype), requires_grad)
    if profile == "small_magnitude":
        profiled = (_rand_like_real(value) * 2 - 1) * 1e-3
        return _with_requires_grad(profiled.to(value.dtype), requires_grad)
    if profile == "large_magnitude":
        scale = 5 if value.dtype in (torch.float16, torch.bfloat16) else 10
        profiled = (_rand_like_real(value) * 2 - 1) * scale
        return _with_requires_grad(profiled.to(value.dtype), requires_grad)
    if profile == "near_one":
        profiled = 1 + (_rand_like_real(value) * 2 - 1) * 1e-2
        return _with_requires_grad(profiled.to(value.dtype), requires_grad)
    if profile == "sparse_first" and tensor_index == 0:
        mask = torch.rand(value.shape, device=value.device) < 0.8
        profiled = _randn_like_real(value)
        profiled = profiled.masked_fill(mask, 0)
        return _with_requires_grad(profiled.to(value.dtype), requires_grad)
    return value


def _apply_value_profile(inputs, profile):
    tensor_index = 0
    profiled_inputs = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            profiled_inputs.append(_profile_tensor(value, profile, tensor_index))
            tensor_index += 1
        else:
            profiled_inputs.append(value)
    return profiled_inputs


def _get_input_trials(context):
    if "get_input_groups" in context:
        return context["get_input_groups"]()
    trials = []
    get_inputs = context["get_inputs"]
    for trial in range(num_correct_trials):
        profile = VALUE_PROFILES[trial % len(VALUE_PROFILES)]
        trials.append(_apply_value_profile(get_inputs(), profile))
    return trials


def _tensor_tolerances(ref_output, new_output):
    dtypes = {ref_output.dtype, new_output.dtype}
    dtype_names = {str(dtype) for dtype in dtypes}
    if any("float8" in name for name in dtype_names):
        return 1e-2, 1e-2
    if torch.float16 in dtypes or torch.bfloat16 in dtypes:
        return 1e-3, 1e-3
    return 1e-4, 1e-4


def _compare_outputs(ref_output, new_output):
    if isinstance(ref_output, torch.Tensor) and isinstance(new_output, torch.Tensor):
        if ref_output.shape != new_output.shape:
            return f"[FAIL] Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
        ref_is_numeric = torch.is_floating_point(ref_output) or torch.is_complex(ref_output)
        new_is_numeric = torch.is_floating_point(new_output) or torch.is_complex(new_output)
        if ref_output.dtype == torch.bool or new_output.dtype == torch.bool or not ref_is_numeric or not new_is_numeric:
            if not torch.equal(ref_output, new_output):
                return "[FAIL] Output mismatch"
        else:
            atol, rtol = _tensor_tolerances(ref_output, new_output)
            if not torch.allclose(ref_output, new_output, atol=atol, rtol=rtol):
                return "[FAIL] Output mismatch"
        return None

    if isinstance(ref_output, (tuple, list)) and isinstance(new_output, (tuple, list)):
        if len(ref_output) != len(new_output):
            return f"[FAIL] Output length mismatch: Expected {len(ref_output)}, got {len(new_output)}"
        for ref_item, new_item in zip(ref_output, new_output):
            feedback = _compare_outputs(ref_item, new_item)
            if feedback is not None:
                return feedback
        return None

    if ref_output != new_output:
        return f"[FAIL] Output mismatch: Expected {ref_output}, got {new_output}"
    return None

    
def execute_template(synchronize, device, context):
    correctness = True
    correctness_information = ''

    get_init_inputs = context['get_init_inputs']
    Model = context['Model']
    ModelNew = context['ModelNew']
        
    try:
        init_inputs = get_init_inputs()
        init_inputs = _to_device(init_inputs, device)
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            original_model = Model(*init_inputs).to(device)
            synchronize(device=device)
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device)
            synchronize(device=device)
        with torch.no_grad():
            for inputs in _get_input_trials(context):
                inputs = _to_device(inputs, device)
                synchronize(device=device)
                ref_output = original_model(*inputs)       
                synchronize(device=device)
                new_output = custom_model(*inputs)
                synchronize(device=device) # ensure all GPU operations are completed before checking results
                feedback = _compare_outputs(ref_output, new_output)
                if feedback is not None:
                    correctness = False
                    correctness_information = feedback
                    break
    except Exception as e:
        print('[FAIL] runtime error when evaluating correctness')
        correctness = False
        correctness_information = f"[FAIL] {str(e)}"
        return correctness, correctness_information

    return correctness, correctness_information
