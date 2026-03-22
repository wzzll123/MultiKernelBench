import re, os, json
from utils.utils import get_ref_src_path
from backends.backend_registry import BACKEND_REGISTRY
import torch
import importlib
import numpy as np
from dataset import dataset
from config import temperature, top_p, num_perf_trials

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code_block = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code_block = code_block[len(code_type) :].strip()
        return code_block

    return None

def eval_single(response_txt:str, op, language):
    # Try to dynamically import the backend if it's not yet registered
    if language not in BACKEND_REGISTRY:
        try:
            importlib.import_module(f"backends.{language}_backend")
        except ImportError as e:
            raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e
    backend = BACKEND_REGISTRY.get(language)
    if backend is None:
        raise ValueError(f"Unsupported language/platform: {language}")
    
    hardware = backend.get_hardware_name()

    result = {'compiled': False, 'correctness': False, 'performance': None, 'hardware':hardware}
    generated_code = extract_first_code(response_txt, ['python', 'cpp'])
    if generated_code is None:
        generated_code = response_txt
    compiled, compile_info = backend.compile(generated_code, op)
    if not compiled:
        result['compile_info'] = compile_info
        return result
    result['compiled'] = True
    ref_src_path = get_ref_src_path(op)
    with open(ref_src_path, 'r') as f:
        ref_src = f.read()
    correctness, info = backend.correctness_execution(ref_src)
    if not correctness:
        result['correctness_info'] = info
        return result
    result['correctness'] = True
    elapsed_times = backend.time_execution()
    result['performance'] = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }
    backend.cleanup()
    return result

def eval_all(out_dir, language, op_tested=dataset.keys()):
    result = {}
    
    for op in op_tested:
        print(f"[INFO] eval op {op}")
        with open(os.path.join(out_dir, f'{op}.txt'), 'r') as saved_log:
            response_txt = saved_log.read()
        result[op] = eval_single(response_txt, op, language)
        
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    
if __name__ == '__main__':
    runs = 1
    model = 'deepseek-chat'
    language = 'cuda'
    op_tested = list(dataset.keys())
    op_tested = ['ltsm_hn', 'conv3d_leaky_relu_sum_clamp_gelu','square_matrix_multiplication','l2_norm','adam','sgd']
    select_shot = False
    for run in range(runs):
        if not select_shot:
            out_dir = f'output/{language}/add_shot/{temperature}-{top_p}/{model}/run{run}'
        else:
            out_dir = f'output/{language}/selected_shot/{temperature}-{top_p}/{model}/run{run}'
        eval_all(out_dir, language, op_tested)
