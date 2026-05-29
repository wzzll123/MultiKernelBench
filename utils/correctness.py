import torch

from config import num_correct_trials, project_root_path, seed_num

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def _to_device(values, device):
    return [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in values
    ]


def _get_input_trials(context):
    if "get_input_groups" in context:
        return context["get_input_groups"]()
    return [context["get_inputs"]() for _ in range(num_correct_trials)]


def _compare_outputs(ref_output, new_output):
    if isinstance(ref_output, torch.Tensor) and isinstance(new_output, torch.Tensor):
        if ref_output.shape != new_output.shape:
            return f"[FAIL] Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
        if ref_output.dtype == torch.bool or new_output.dtype == torch.bool:
            if not torch.equal(ref_output, new_output):
                return "[FAIL] Output mismatch"
        elif not torch.allclose(ref_output, new_output, atol=1e-04, rtol=1e-04):
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
