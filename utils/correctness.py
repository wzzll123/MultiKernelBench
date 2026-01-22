import torch

from config import num_correct_trials, project_root_path, seed_num

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)

    
def execute_template(synchronize, device, context):
    correctness = True
    correctness_information = ''

    get_inputs = context['get_inputs']
    get_init_inputs = context['get_init_inputs']
    Model = context['Model']
    ModelNew = context['ModelNew']
        
    try:
        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            original_model = Model(*init_inputs).to(device)
            synchronize(device=device)
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device)
            synchronize(device=device)
        with torch.no_grad():
            for trial in range(num_correct_trials):
                inputs = get_inputs()
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                synchronize(device=device)
                ref_output = original_model(*inputs)       
                synchronize(device=device)
                new_output = custom_model(*inputs)
                synchronize(device=device) # ensure all GPU operations are completed before checking results
                feedback = None
                if ref_output.shape != new_output.shape:
                    feedback = f"[FAIL] Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                elif not torch.allclose(ref_output, new_output, atol=1e-04, rtol=1e-04):
                    feedback = f"[FAIL] Output mismatch"
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
