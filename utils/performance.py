from config import num_perf_trials, num_warmup
import torch

def _to_device(values, device):
    return [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in values
    ]

def _get_perf_inputs(context):
    if "get_input_groups" in context:
        input_groups = context["get_input_groups"]()
        if not input_groups:
            raise ValueError("get_input_groups() returned no inputs")
        return input_groups[0]
    return context['get_inputs']()

def time_execution_event_template(context, device, synchronize, event_class, eval_target):
    get_init_inputs = context['get_init_inputs']
    generated_elapsed_times = []
    ModelNew = context[eval_target]
    inputs = _to_device(_get_perf_inputs(context), device)
    init_inputs = get_init_inputs()
    init_inputs = _to_device(init_inputs, device)
    with torch.no_grad():
        custom_model = ModelNew(*init_inputs).to(device)
        def internel_eval(kernel_fn, elapsed_times):
            for _ in range(num_warmup):
                kernel_fn(*inputs)
                synchronize(device=device)
            for trail in range(num_perf_trials):
                start_event = event_class(enable_timing=True)
                end_event = event_class(enable_timing=True)
                start_event.record()
                kernel_fn(*inputs)
                end_event.record()
                # Synchronize to ensure the events have completed
                synchronize(device=device)
                # Calculate the elapsed time in milliseconds
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_times.append(elapsed_time_ms)
        internel_eval(custom_model, generated_elapsed_times)
    return generated_elapsed_times
